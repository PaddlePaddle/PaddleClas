# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HTTP and filesystem regression tests; inference dependencies are mocked.

Run with: python -m unittest discover -s tests -p test_shitu_index_security.py
Requires fastapi, httpx, uvicorn, numpy and urllib3, but no model weights or GPU.
"""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

SERVICE_DIR = Path(
    __file__).resolve().parents[1] / "deploy/shitu_index_manager"


def load_server(source=None):
    dependencies = {
        name: mock.MagicMock()
        for name in (
            "PyQt5",
            "mod.mainwindow",
            "paddleclas",
            "paddleclas.deploy",
            "paddleclas.deploy.utils",
            "paddleclas.deploy.python",
            "paddleclas.deploy.python.predict_rec",
            "faiss",
            "cv2",
        )
    }
    spec = importlib.util.spec_from_file_location("shitu_security_server",
                                                  SERVICE_DIR / "server.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, dependencies), mock.patch.object(
            sys, "path", [str(SERVICE_DIR)] + sys.path):
        if source is None:
            spec.loader.exec_module(module)
        else:
            exec(compile(source, str(SERVICE_DIR / "server.py"), "exec"),
                 module.__dict__)
    return module


class IndexSecurityTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.allowed = self.base / "galleries"
        self.allowed.mkdir()
        self.gallery = self.allowed / "gallery"
        self.gallery.mkdir()
        (self.gallery / "images").mkdir()
        self.image_list = self.gallery / "image_list.txt"
        self.image_list.write_text("images/item.jpg item label\n",
                                   encoding="utf-8")
        self.secret = self.base / "secret.txt"
        self.marker = "PRIVATE_TEST_MARKER"
        self.secret.write_text(self.marker + "\n", encoding="utf-8")
        self.env = mock.patch.dict(
            os.environ,
            {
                "PADDLECLAS_INDEX_ROOT": str(self.allowed),
                "PADDLECLAS_INDEX_TOKEN": "test-only-token",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.server = load_server()
        self.manager = self.server.ShiTuIndexManager({
            "Global": {
                "batch_size": 1
            },
            "IndexProcess": {
                "embedding_size": 2
            }
        })
        self.server.manager = self.manager
        self.client = TestClient(self.server.app)
        self.addCleanup(self.client.close)
        self.headers = {"Authorization": "Bearer test-only-token"}

    def request(
        self,
        endpoint,
        image_list_path="image_list.txt",
        root=None,
        headers=None,
        **extra,
    ):
        params = {
            "index_root_path": str(self.gallery if root is None else root),
            "image_list_path": str(image_list_path),
        }
        params.update(extra)
        return self.client.get(
            endpoint,
            params=params,
            headers=self.headers if headers is None else headers,
        )

    def error(self, response):
        self.assertEqual(response.status_code, 200)
        result = response.json()
        if isinstance(result, str):
            result = json.loads(result)
        return result["error_message"]

    def assert_blocked(self, path, root=None):
        for endpoint in ("/open_index", "/new_index", "/update_index"):
            with self.subTest(endpoint=endpoint, path=str(path)):
                with mock.patch(
                        "builtins.open",
                        side_effect=AssertionError(
                            "An invalid request must not read a file"),
                ) as file_open:
                    response = self.request(endpoint, path, root=root)
                self.assertTrue(self.error(response))
                self.assertNotIn(self.marker, response.text)
                file_open.assert_not_called()

    def test_anonymous_requests_are_rejected_before_file_access(self):
        for endpoint in ("/open_index", "/new_index", "/update_index"):
            with self.subTest(endpoint=endpoint), mock.patch(
                    "builtins.open") as file_open:
                self.assertEqual(
                    self.request(endpoint, self.secret,
                                 headers={}).status_code, 401)
                file_open.assert_not_called()

    def test_wrong_token_is_rejected(self):
        for header in ("Bearer wrong-token", "Basic test-only-token"):
            self.assertEqual(
                self.request("/open_index", headers={
                    "Authorization": header
                }).status_code,
                401,
            )

    def test_unconfigured_auth_fails_closed(self):
        self.server.index_token = None
        self.assertEqual(self.request("/open_index").status_code, 503)

    def test_absolute_path_escape(self):
        self.assert_blocked(self.secret)

    def test_parent_path_escape(self):
        self.assert_blocked("../../secret.txt")

    def test_client_cannot_choose_an_unrestricted_root(self):
        self.assert_blocked("secret.txt", root=self.base)

    def test_directory_prefix_is_not_containment(self):
        sibling = self.base / "galleries-extra"
        sibling.mkdir()
        self.assert_blocked("image_list.txt", root=sibling)

    def test_replacing_allowed_root_symlink_cannot_rebase_policy(self):
        self.allowed.rename(self.base / "original-galleries")
        self.allowed.symlink_to(self.base, target_is_directory=True)
        self.assert_blocked("secret.txt", root=self.allowed)

    def test_concurrent_requests_do_not_interleave_manager_operations(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        import time

        active = 0
        peak = 0
        counter_lock = threading.Lock()

        def operation(*args, **kwargs):
            nonlocal active, peak
            with counter_lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.02)
                return ""
            finally:
                with counter_lock:
                    active -= 1

        with mock.patch.object(self.manager,
                               "open_index",
                               side_effect=operation):
            with ThreadPoolExecutor(max_workers=8) as executor:
                responses = list(
                    executor.map(lambda _: self.request("/open_index"),
                                 range(8)))
        self.assertEqual(peak, 1)
        for response in responses:
            self.assertEqual(self.error(response), "")

    def test_list_symlink_escape(self):
        (self.gallery / "linked.txt").symlink_to(self.secret)
        self.assert_blocked("linked.txt")

    def test_gallery_symlink_escape(self):
        linked = self.allowed / "linked-gallery"
        linked.symlink_to(self.base, target_is_directory=True)
        self.assert_blocked("secret.txt", root=linked)

    def test_index_directory_symlink_escape(self):
        (self.gallery / "index").symlink_to(self.base,
                                            target_is_directory=True)
        self.assert_blocked("image_list.txt")

    def test_invalid_list_never_echoes_contents(self):
        self.image_list.write_text(self.marker + "\n", encoding="utf-8")
        self.manager.root_path = str(self.gallery)
        self.manager.index = object()
        self.manager.id_map = {0: "item"}
        self.manager.features = {"image_ids": ["item"]}
        for endpoint in ("/open_index", "/new_index", "/update_index"):
            with self.subTest(endpoint=endpoint):
                response = self.request(endpoint)
                self.assertTrue(self.error(response))
                self.assertNotIn(self.marker, response.text)

    def test_nul_separated_single_line_is_not_echoed(self):
        self.image_list.write_bytes(b"PRIVATE_TEST_MARKER\0TOKEN=test-only\0")
        response = self.request("/open_index")
        self.assertTrue(self.error(response))
        self.assertNotIn(self.marker, response.text)
        self.assertNotIn("TOKEN=test-only", response.text)

    def test_arbitrary_backend_exception_is_not_echoed(self):
        for endpoint, method in (
            ("/open_index", "open_index"),
            ("/new_index", "create_index"),
            ("/update_index", "update_index"),
        ):
            with self.subTest(endpoint=endpoint), mock.patch.object(
                    self.manager, method,
                    side_effect=RuntimeError(self.marker)):
                response = self.request(endpoint)
                self.assertTrue(self.error(response))
                self.assertNotIn(self.marker, response.text)

    def test_image_path_escape_is_rejected(self):
        self.image_list.write_text(str(self.secret) + " label\n",
                                   encoding="utf-8")
        with self.assertRaises(ValueError):
            self.manager._split_datafile(str(self.image_list),
                                         str(self.gallery))

    def test_image_symlink_escape_is_rejected(self):
        (self.gallery / "images/item.jpg").symlink_to(self.secret)
        with self.assertRaises(ValueError):
            self.manager._split_datafile(str(self.image_list),
                                         str(self.gallery))

    def test_valid_relative_and_absolute_list_paths(self):
        for path in ("image_list.txt", str(self.image_list)):
            with self.subTest(path=path):
                response = self.request("/open_index", path)
                self.assertEqual(
                    self.error(response),
                    "File not exist: features.pkl, vector.index, id_map.pkl",
                )

    def test_valid_list_parsing_preserves_labels_and_ids(self):
        paths, docs, ids = self.manager._split_datafile(
            str(self.image_list), str(self.gallery))
        self.assertEqual(paths, [str(self.gallery / "images/item.jpg")])
        self.assertEqual(docs, ["images/item.jpg item label"])
        self.assertEqual(ids, ["item"])

    def test_create_open_and_update_with_mocked_inference(self):
        self.manager._cal_featrue = mock.Mock(return_value=np.ones((1, 2)))
        self.assertEqual(self.error(self.request("/new_index")), "")
        self.assertEqual(self.manager.root_path, str(self.gallery))
        self.assertEqual(self.manager.features["image_ids"], ["item"])
        # FAISS is mocked; create its artifact so open follows the load branch.
        (self.gallery / "index/vector.index").touch()
        self.assertEqual(self.error(self.request("/open_index")), "")
        self.assertEqual(self.error(self.request("/update_index")), "")

    def test_create_reports_missing_list(self):
        self.assertEqual(
            self.error(self.request("/new_index", "missing.txt")),
            "Image list file does not exist",
        )

    def test_update_requires_an_open_index(self):
        self.assertEqual(
            self.error(self.request("/update_index")),
            "Failed. Please create or open index first",
        )

    def test_pickle_symlink_escape_is_rejected(self):
        self.manager.root_path = str(self.gallery)
        (self.gallery / "features.pkl").symlink_to(self.secret)
        with mock.patch("builtins.open") as file_open:
            with self.assertRaises(ValueError):
                self.manager._load_pickle(str(self.gallery / "features.pkl"))
            file_open.assert_not_called()

    def load_http_client(self):
        spec = importlib.util.spec_from_file_location(
            "shitu_security_client", SERVICE_DIR / "mod/index_http_client.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pool = mock.Mock()

        def request(method, url, headers):
            response = self.client.request(method, url, headers=headers)
            return SimpleNamespace(data=response.content,
                                   status=response.status_code)

        pool.request.side_effect = request
        with mock.patch.object(module.urllib3,
                               "PoolManager",
                               return_value=pool):
            return module.IndexHttpClient("testserver", 8000)

    def test_client_sends_token_and_decodes_server_response(self):
        client = self.load_http_client()
        self.assertEqual(
            client.open_index(str(self.gallery), "image_list.txt"),
            "File not exist: features.pkl, vector.index, id_map.pkl",
        )

    def test_client_handles_authentication_error(self):
        with mock.patch.dict(os.environ,
                             {"PADDLECLAS_INDEX_TOKEN": "wrong-token"}):
            client = self.load_http_client()
        self.assertEqual(
            client.open_index(str(self.gallery), "image_list.txt"),
            "Unauthorized")

    def test_client_without_token_handles_authentication_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            client = self.load_http_client()
        self.assertEqual(
            client.open_index(str(self.gallery), "image_list.txt"),
            "Unauthorized")

    def test_launcher_shares_generated_token_with_client(self):
        import runpy

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
                sys.modules, {"psutil": mock.MagicMock()}), mock.patch.object(
                    sys, "argv",
                    ["index_manager.py", "-c", "config with spaces.yaml"
                     ]), mock.patch("subprocess.Popen") as popen:
            runpy.run_path(str(SERVICE_DIR / "index_manager.py"),
                           run_name="__main__")
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(popen.call_args_list[0].args[0][0], sys.executable)
        self.assertEqual(popen.call_args_list[1].args[0][0], sys.executable)
        self.assertIn("config with spaces.yaml",
                      popen.call_args_list[0].args[0])
        server_env = popen.call_args_list[0].kwargs["env"]
        client_env = popen.call_args_list[1].kwargs["env"]
        self.assertEqual(server_env, client_env)
        self.assertGreaterEqual(len(server_env["PADDLECLAS_INDEX_TOKEN"]), 32)
        self.assertEqual(server_env["PADDLECLAS_INDEX_ROOT"], os.getcwd())


if __name__ == "__main__":
    unittest.main()
