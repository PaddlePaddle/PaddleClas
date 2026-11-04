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

"""Real image/index integration tests with deterministic feature extraction.

FAISS, OpenCV, disk artifacts and HTTP handlers are real. Only the Paddle
predictor and desktop imports are mocked. Install faiss-cpu and
opencv-python-headless in addition to the security-test dependencies.
"""

import unittest
import os

import numpy as np

try:
    import cv2
    import faiss
except ImportError:
    cv2 = None
    faiss = None

import test_shitu_index_security as security_tests


class DeterministicPredictor:
    def predict(self, images):
        features = np.asarray(
            [list(image.mean(axis=(0, 1))) + [1.0] for image in images],
            dtype=np.float32,
        )
        return features / np.linalg.norm(features, axis=1, keepdims=True)


@unittest.skipIf(faiss is None or cv2 is None, "Requires faiss-cpu and OpenCV")
class RealIndexIntegrationTests(unittest.TestCase):
    def setUp(self):
        previous_threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)
        self.addCleanup(faiss.omp_set_num_threads, previous_threads)
        self.case = security_tests.IndexSecurityTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.case.server.faiss = faiss
        self.case.server.cv2 = cv2
        self.case.manager.config["IndexProcess"]["embedding_size"] = 4
        self.case.manager.config["Global"]["batch_size"] = 2
        self.case.manager.predictor = DeterministicPredictor()
        for i, name in enumerate(("a", "b", "c", "d"), 1):
            image = np.full((16, 16, 3), i * 40, dtype=np.uint8)
            self.assertTrue(
                cv2.imwrite(str(self.case.gallery / "images" / (name + ".jpg")), image)
            )

    def set_list(self, names):
        self.case.image_list.write_text(
            "".join("images/{}.jpg {}\n".format(name, name) for name in names),
            encoding="utf-8",
        )

    def assert_success(self, endpoint, **kwargs):
        self.assertEqual(self.case.error(self.case.request(endpoint, **kwargs)), "")

    def test_real_create_restart_add_remove_add_again(self):
        self.set_list(["a", "b"])
        self.assert_success("/new_index")
        self.assertEqual(self.case.manager.index.ntotal, 2)
        artifact = self.case.gallery / "index/vector.index"
        self.assertEqual(faiss.read_index(str(artifact)).ntotal, 2)

        restarted = self.case.server.ShiTuIndexManager(self.case.manager.config)
        restarted.predictor = DeterministicPredictor()
        self.case.manager = restarted
        self.case.server.manager = restarted
        self.assert_success("/open_index")
        self.assertEqual(restarted.index.ntotal, 2)

        self.set_list(["b", "c"])
        self.assert_success("/update_index")
        self.assertEqual(set(restarted.features["image_ids"]), {"b", "c"})
        self.set_list(["b", "c", "d"])
        self.assert_success("/update_index")
        self.assertEqual(restarted.index.ntotal, 3)
        self.assertEqual(faiss.read_index(str(artifact)).ntotal, 3)
        self.assert_success("/open_index")
        self.assertEqual(set(restarted.features["image_ids"]), {"b", "c", "d"})

    def test_switching_to_missing_index_does_not_reuse_previous_gallery(self):
        self.set_list(["a", "b"])
        self.assert_success("/new_index")
        other = self.case.allowed / "other"
        other.mkdir()
        (other / "image_list.txt").write_text("images/a.jpg a\n", encoding="utf-8")
        response = self.case.request("/open_index", root=other)
        self.assertTrue(self.case.error(response))
        self.assertIsNone(self.case.manager.index)
        self.assertIsNone(self.case.manager.id_map)
        self.assertIsNone(self.case.manager.features)
        self.assertEqual(
            self.case.error(self.case.request("/update_index", root=other)),
            "Failed. Please create or open index first",
        )

    def test_failed_rebuild_preserves_loaded_index(self):
        self.set_list(["a", "b"])
        self.assert_success("/new_index")
        old_index = self.case.manager.index
        self.set_list(["missing"])
        self.assertTrue(self.case.error(self.case.request("/new_index", force=True)))
        self.assertIs(self.case.manager.index, old_index)
        self.assertEqual(old_index.ntotal, 2)
        self.set_list(["a", "b"])
        self.assert_success("/update_index")
        artifact = self.case.gallery / "index/vector.index"
        self.assertEqual(faiss.read_index(str(artifact)).ntotal, 2)

    def test_gui_flat_method_and_lowercase_alias(self):
        self.set_list(["a", "b"])
        for method in ("FLAT", "flat", "hnsw32", "IVF"):
            with self.subTest(method=method):
                self.assert_success("/new_index", index_method=method, force=True)
                self.assertEqual(self.case.manager.index.ntotal, 2)
                self.assert_success("/open_index")

    @unittest.skipUnless(
        os.environ.get("PADDLECLAS_TEST_NETWORK") == "1",
        "Set PADDLECLAS_TEST_NETWORK=1 for the loopback HTTP test",
    )
    def test_real_http_client_and_server(self):
        import importlib.util
        import socket
        import threading
        import time

        import httpx
        import uvicorn

        self.set_list(["a", "b"])
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(self.case.server.app, log_level="error"))
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )

        def stop():
            server.should_exit = True
            thread.join(timeout=10)
            listener.close()

        thread.start()
        self.addCleanup(stop)
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(server.started, "The HTTP service did not start")
        spec = importlib.util.spec_from_file_location(
            "shitu_network_client",
            security_tests.SERVICE_DIR / "mod/index_http_client.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        client = module.IndexHttpClient("127.0.0.1", port)
        root = str(self.case.gallery)
        self.assertIsNone(client.new_index("image_list.txt", root))
        self.assertIsNone(client.open_index(root, "image_list.txt"))
        self.set_list(["b", "c"])
        self.assertIsNone(client.update_index("image_list.txt", root))
        self.set_list(["b", "c", "d"])
        self.assertIsNone(client.update_index("image_list.txt", root))
        self.assertEqual(self.case.manager.index.ntotal, 3)
        error = client.open_index(root, str(self.case.secret))
        self.assertTrue(error)
        self.assertNotIn(self.case.marker, error)
        response = httpx.get(
            "http://127.0.0.1:{}/open_index".format(port),
            params={"index_root_path": root, "image_list_path": str(self.case.secret)},
        )
        self.assertEqual(response.status_code, 401)
        self.assertNotIn(self.case.marker, response.text)


if __name__ == "__main__":
    unittest.main()
