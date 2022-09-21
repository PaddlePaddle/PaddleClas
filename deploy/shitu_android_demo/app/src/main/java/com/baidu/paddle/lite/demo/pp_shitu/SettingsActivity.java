/*
 * Copyright (C) 2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.baidu.paddle.lite.demo.pp_shitu;

import android.app.AlertDialog;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.ListPreference;
import android.support.v7.app.ActionBar;
import android.view.View;
import android.widget.Button;
import android.widget.ListView;
import android.widget.Toast;

import com.baidu.paddle.lite.demo.common.AppCompatPreferenceActivity;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class SettingsActivity extends AppCompatPreferenceActivity implements SharedPreferences.OnSharedPreferenceChangeListener {
    ListPreference lpChoosePreInstalledModel = null;
    ListPreference lpLabelPath = null;
    ListPreference lpIndexPath = null;


    List<String> preInstalledModelPaths = null;
    List<String> preInstalledLabelPaths = null;
    List<String> preInstalledIndexDirs = null;
    List<String> preInstalledImagePaths = null;


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(R.xml.settings);
        ActionBar supportActionBar = getSupportActionBar();
        if (supportActionBar != null) {
            supportActionBar.setDisplayHomeAsUpEnabled(true);
        }


        // Initialized pre-installed models
        preInstalledModelPaths = new ArrayList<>();
        preInstalledLabelPaths = new ArrayList<>();
        preInstalledIndexDirs = new ArrayList<>();
        preInstalledImagePaths = new ArrayList<>();

        // Add mobilenet_v1_for_cpu
        preInstalledModelPaths.add(getString(R.string.MODEL_PATH_DEFAULT));
        preInstalledLabelPaths.add(getString(R.string.LABEL_PATH_DEFAULT));
        preInstalledIndexDirs.add(getString(R.string.INDEX_PATH_DEFAULT));
        preInstalledImagePaths.add(getString(R.string.IMAGE_PATH_DEFAULT));

        // Setup UI components
        lpChoosePreInstalledModel =
                (ListPreference) findPreference(getString(R.string.CHOOSE_PRE_INSTALLED_MODEL_KEY));
        String[] preInstalledModelNames = new String[preInstalledModelPaths.size()];
        for (int i = 0; i < preInstalledModelPaths.size(); i++) {
            preInstalledModelNames[i] =
                    preInstalledModelPaths.get(i).substring(preInstalledModelPaths.get(i).lastIndexOf("/") + 1);
        }
        lpChoosePreInstalledModel.setEntries(preInstalledModelNames);
        lpChoosePreInstalledModel.setEntryValues(preInstalledModelPaths.toArray(new String[preInstalledModelPaths.size()]));

        lpLabelPath = (ListPreference) findPreference(getString(R.string.LABEL_PATH_KEY));
        String label_dir = getExternalFilesDir(null) + "/index/";
        File dir = new File(label_dir);
        String[] files = dir.list();
        ArrayList<String> files_ = new ArrayList<>();
        for (int i = 0; i < Objects.requireNonNull(files).length; i++) {
            if (!files[i].endsWith(".txt")) {
                continue;
            }
            files_.add("index/" + files[i]);
            files[i] = label_dir + files[i];
        }
        lpLabelPath.setEntries(files_.toArray(new String[files_.size()]));
        lpLabelPath.setEntryValues(files_.toArray(new String[files_.size()]));

        lpIndexPath = (ListPreference) findPreference(getString(R.string.INDEX_PATH_KEY));
        String index_dir = getExternalFilesDir(null) + "/index/";
        dir = new File(index_dir);
        files = dir.list();
        files_ = new ArrayList<>();
        for (int i = 0; i < Objects.requireNonNull(files).length; i++) {
            if (!files[i].endsWith(".index")) {
                continue;
            }
            files_.add("index/" + files[i]);
            files[i] = index_dir + files[i];
        }
        lpIndexPath.setEntries(files_.toArray(new String[files_.size()]));
        lpIndexPath.setEntryValues(files_.toArray(new String[files_.size()]));
    }

    private void reloadPreferenceAndUpdateUI() {
        SharedPreferences sharedPreferences = getPreferenceScreen().getSharedPreferences();
        String modelPath = "models";
        int modelIdx = lpChoosePreInstalledModel.findIndexOfValue(modelPath);
        if (modelIdx >= 0 && modelIdx < preInstalledModelPaths.size()) {
            lpChoosePreInstalledModel.setSummary(modelPath);
        }

        String labelPath = sharedPreferences.getString(getString(R.string.LABEL_PATH_KEY),
                getString(R.string.LABEL_PATH_DEFAULT));
        String indexPath = sharedPreferences.getString(getString(R.string.INDEX_PATH_KEY),
                getString(R.string.INDEX_PATH_DEFAULT));

        lpLabelPath.setSummary(labelPath);
        lpIndexPath.setSummary(indexPath);
    }

    @Override
    protected void onResume() {
        super.onResume();
        getPreferenceScreen().getSharedPreferences().registerOnSharedPreferenceChangeListener(this);
        reloadPreferenceAndUpdateUI();
    }

    @Override
    protected void onPause() {
        super.onPause();
        getPreferenceScreen().getSharedPreferences().unregisterOnSharedPreferenceChangeListener(this);
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
        reloadPreferenceAndUpdateUI();
    }
}
