package com.example.application_3;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Typeface;
import android.os.Bundle;
import android.util.Base64;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.Button;
import android.net.Uri;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.res.ResourcesCompat;

import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class Result extends AppCompatActivity {
    @SuppressLint("MissingInflatedId")

    private String costTime;
    private String Lan = "http://192.168.137.1:8000/";
    @Override


    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //这个是获取布局文件的，这里是你下一个页面的布局文件//注意这个是跳转界面的不能设置错，应该是第一个
        setContentView(R.layout.m_result);
        Intent intent = getIntent();
        Bundle extras = intent.getExtras();
        String[] reslut = (String[]) extras.getSerializable("reslut");
        String id = reslut[0];
        String confidence = reslut[1];
        String runTime = reslut[2];
        costTime = runTime;
        if (confidence.length() >= 6) {
            confidence = confidence.substring(0, 6);
        } else {
            // 如果字符串长度不足六个字符，使用整个字符串
            confidence = confidence.substring(0, confidence.length());
        }
        float con = Float.parseFloat(confidence);
        con = con*100;
        String conStr = String.valueOf(con);

        String classname[] = new String[4];
        // ['Ao Mango', 'GuiQi Mango', 'Jinhuang Mango', 'Tainong Mango']

        classname[0] = "Ao Mango";
        classname[1] = "GuiQi Mango";
        classname[2] = "Jinhuang Mango";
        classname[3] = "Tainong Mango";

        String introduce[] = new String[4];


        introduce[0]="Australian mangoes are oval-shaped with a color transitioning from green to orange. They offer a sweet taste paired with a rich mango aroma, making them distinct and delightful.";
        introduce[1]="Guifei mangoes, also known as Guiqi mangoes, have an elongated shape with green to yellow-orange skin. They are known for their sweet taste, minimal fiber, and rich, juicy flavor, making them a favorite among mango enthusiasts.";
        introduce[2]="Hongjinhuang mangoes are distinguished by their large size and red-orange skin. They boast a sweet and juicy flavor, with a smooth texture and minimal fibers, making them a sought-after variety among mango lovers.";
        introduce[3]="Tainong mangoes are characterized by their elongated shape and vibrant red or orange skin. They offer a sweet and aromatic flavor profile with a smooth, fiber-free texture, making them a highly desirable choice among mango aficionados.";



        int Id = Integer.parseInt(id);
        Typeface typeface = ResourcesCompat.getFont(this, R.font.micross);

        TextView textView = findViewById(R.id.mango_name); // 获取TextView实例
        textView.setTypeface(typeface);
        textView.setText("Result:"+classname[Id]); // 设置TextView的文本内容

        TextView text_con = findViewById(R.id.con);
        text_con.setTypeface(typeface);
        text_con.setText("Accuracy:"+conStr+"%");

        RoundImageView roundImageView;
        roundImageView = findViewById(R.id.View_1); // 通过ID获取控件
        switch (Id){
            case 0:
                roundImageView.setImageResource(R.drawable.ao_mango);
                break;
            case 1:
                roundImageView.setImageResource(R.drawable.guiqi_mango);
                break;
            case 2:
                roundImageView.setImageResource(R.drawable.jinhuang_mango);
                break;
            case 3:
                roundImageView.setImageResource(R.drawable.tainong_mango);
                break;
        }

        TextView textIn = findViewById(R.id.mango_in); // 获取TextView实例
        textIn.setTypeface(typeface);
        textIn.setText("Introduce:"+introduce[Id]); // 设置TextView的文本内容

        TextView textRT = findViewById(R.id.runTime); // 获取TextView实例
        textRT.setTypeface(typeface);
        textRT.setText("runTime:"+runTime+"ms"); // 设置TextView的文本内容

        Button httpButton;
        httpButton = (Button) findViewById(R.id.http_button);

        String http[] = new String[4];
        http[0] = "https://baike.baidu.com/item/%E6%BE%B3%E8%8A%92/3925195?fr=ge_ala";
        http[1] = "https://baike.baidu.com/item/%E6%A1%82%E4%B8%83%E8%8A%92%E6%9E%9C/17928766?fr=ge_ala";
        http[2] = "https://baike.baidu.com/item/%E7%BA%A2%E9%87%91%E7%85%8C/16030028?fr=ge_ala";
        http[3] = "https://baike.baidu.com/item/%E5%8F%B0%E5%86%9C%E8%8A%92/62640030?fr=ge_ala";


        // 为按钮添加点击事件监听器
        httpButton.setOnClickListener(v -> {
            String url = http[Id]; // 替换为你的目标URL
            Intent browserIntent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
            startActivity(browserIntent);
        });
        sendGetRequest();




    }

    private void sendGetRequest() {

        String t = costTime;
        OkHttpClient client = new OkHttpClient();
        // 将参数添加到URL中
        String url = Lan + "process/recordtime/?time=" + t;

        Request request = new Request.Builder()
                .url(url) // 设置请求的URL
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
                            //handleResponseData(responseData);
                            //Toast.makeText(Result.this, "请求成功：" + responseData, Toast.LENGTH_SHORT).show();
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




}
