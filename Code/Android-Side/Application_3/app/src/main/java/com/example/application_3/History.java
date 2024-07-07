package com.example.application_3;

import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Base64;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.bumptech.glide.Glide;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class History extends AppCompatActivity {
    private ListView mListView;
    private List<String> mStringList;
    private List<Bitmap> mBitmapList;
    private CustomAdapter mArrayAdapter;

    private TextView ClearText;

    private String Lan = "http://192.168.137.1:8000/";
    private String Ngrok = "https://frankly-content-platypus.ngrok-free.app/";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.history);

        ImageView loadView = findViewById(R.id.hisLoadView);

        Button clearButton = findViewById(R.id.clearButton);
        clearButton.setVisibility(View.GONE);
        ClearText = findViewById(R.id.clearText);
        ClearText.setVisibility(View.GONE);

        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ImageView loadView = findViewById(R.id.hisLoadView);
                loadView.setVisibility(View.VISIBLE);
                Button clearButton = findViewById(R.id.clearButton);
                clearButton.setVisibility(View.GONE);
                ClearText = findViewById(R.id.clearText);
                ClearText.setVisibility(View.GONE);
                mListView = findViewById(R.id.lv);
                mListView.setVisibility(View.GONE);
                clear();
            }
        });

        // 加载 GIF 动图
        Glide.with(this)
                .load(R.drawable.load)
                .into(loadView);

        mListView = findViewById(R.id.lv);
        mStringList = new ArrayList<>();
        mBitmapList = new ArrayList<>();
        mArrayAdapter = new CustomAdapter(this, R.layout.list_item_with_image, mStringList, mBitmapList);
        mListView.setAdapter(mArrayAdapter);

        mListView.setOnItemLongClickListener(new AdapterView.OnItemLongClickListener() {
            @Override
            public boolean onItemLongClick(AdapterView<?> parent, View view, int position, long id) {
                // 处理长按事件的代码
                showDeleteDialog(position);
                return true; // 返回true表示已处理长按事件
            }
        });

        sendGetRequest();
    }

    private void sendGetRequest() {
        OkHttpClient client = new OkHttpClient();
        Request request = new Request.Builder()
                .url(Lan + "process/info/") // 设置请求的URL
                .get() // 指定使用GET请求方法
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        // 处理请求失败的情况，例如显示错误信息或重试请求
                    }
                });
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    final String responseData = response.body().string();

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            // 在主线程更新UI
                            handleResponseData(responseData);
                        }
                    });
                } else {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            // 处理请求失败的情况，例如显示错误信息或重试请求
                        }
                    });
                }
            }
        });
    }

    private void deleteRequest(String id) {
        OkHttpClient client = new OkHttpClient();
        String deleteUrl = Lan + "process/delete/?id=" + id; // 包含要删除的项的唯一标识

        Request request = new Request.Builder()
                .url(deleteUrl) // 设置包含要删除的项 ID 的 URL
                .get() // 指定使用GET请求方法
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        // 处理请求失败的情况，例如显示错误信息或重试请求
                    }
                });
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    final String responseData = response.body().string();

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            // 在主线程更新UI
                            handleResponseData(responseData);
                        }
                    });
                } else {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            // 处理请求失败的情况，例如显示错误信息或重试请求
                        }
                    });
                }
            }
        });
    }

    private void clear() {
        OkHttpClient client = new OkHttpClient();
        String deleteUrl = Lan + "process/clear/"; // 包含要删除的项的唯一标识

        Request request = new Request.Builder()
                .url(deleteUrl) // 设置包含要删除的项 ID 的 URL
                .get() // 指定使用GET请求方法
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        // 处理请求失败的情况，例如显示错误信息或重试请求
                    }
                });
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    final String responseData = response.body().string();

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            // 在主线程更新UI
                            // handleResponseData(responseData);

                            mStringList.clear();
                            mBitmapList.clear();
                            mArrayAdapter.notifyDataSetChanged();

                            ImageView loadView = findViewById(R.id.hisLoadView);
                            loadView.setVisibility(View.GONE);
                            Button clearButton = findViewById(R.id.clearButton);
                            clearButton.setVisibility(View.VISIBLE);
                            ClearText = findViewById(R.id.clearText);
                            ClearText.setVisibility(View.VISIBLE);

                            mListView = findViewById(R.id.lv);
                            mListView.setVisibility(View.VISIBLE);
                        }
                    });
                } else {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            // 处理请求失败的情况，例如显示错误信息或重试请求
                        }
                    });
                }
            }
        });
    }

    private void handleResponseData(String responseData) {
        Gson gson = new Gson();
        JsonObject jsonObject = gson.fromJson(responseData, JsonObject.class);
        int resultSize = jsonObject.get("result_size").getAsInt();
        JsonArray items = jsonObject.getAsJsonArray("items");

        String[][] info = new String[resultSize][5];

        for (int i = 0; i < resultSize; i++) {
            JsonObject item = items.get(i).getAsJsonObject();
            info[i][0] = item.get("time").getAsString();
            String confidence = item.get("confidence").getAsString();

            if (confidence.length() >= 6) {
                confidence = confidence.substring(0, 6);
            } else {
                confidence = confidence.substring(0, confidence.length());
            }
            float con = Float.parseFloat(confidence);
            con = con * 100;
            info[i][1] = String.valueOf(con);
            info[i][2] = item.get("runTime").getAsString();
            info[i][3] = item.get("image").getAsString();
            info[i][4] = item.get("id").getAsString();
        }

        mStringList.clear();
        mBitmapList.clear();
        String classname[] = new String[4];
        // ['Ao Mango', 'GuiQi Mango', 'Jinhuang Mango', 'Tainong Mango']

        classname[0] = "Ao Mango";
        classname[1] = "GuiQi Mango";
        classname[2] = "Jinhuang Mango";
        classname[3] = "Tainong Mango";
        for (int i = 0; i < resultSize; i++) {
            byte[] bytes = Base64.decode(info[i][3], Base64.DEFAULT);
            Bitmap image = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
            String id = info[i][4];
            int num = Integer.parseInt(id);
            mStringList.add("\n" +
                    "Mango Name:"+classname[num] + "\n" +
                    "Time: " + info[i][0] + "\n" +
                    "Confidence: " + info[i][1] + "%" + "\n" +
                    "Run time: " + info[i][2] + "ms" + "\n"
            );
            mBitmapList.add(image);
        }

        ImageView loadView = findViewById(R.id.hisLoadView);
        loadView.setVisibility(View.GONE);
        Button clearButton = findViewById(R.id.clearButton);
        clearButton.setVisibility(View.VISIBLE);
        ClearText = findViewById(R.id.clearText);
        ClearText.setVisibility(View.VISIBLE);

        mListView = findViewById(R.id.lv);
        mListView.setVisibility(View.VISIBLE);
        mArrayAdapter.notifyDataSetChanged();
    }

    private void showDeleteDialog(final int position) {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Are you sure to delete the test record ?" );
        builder.setPositiveButton("Yes", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                ImageView loadView = findViewById(R.id.hisLoadView);
                loadView.setVisibility(View.VISIBLE);
                Button clearButton = findViewById(R.id.clearButton);
                clearButton.setVisibility(View.GONE);
                mListView = findViewById(R.id.lv);
                mListView.setVisibility(View.GONE);

                deleteItemAndSendRequest(position);
            }
        });

        builder.setNegativeButton("No", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                // 用户取消删除, 那么什么都不做
            }
        });

        builder.create().show();
    }

    private void deleteItemAndSendRequest(final int position) {
        String id = String.valueOf(position);
        deleteRequest(id);
    }
}
