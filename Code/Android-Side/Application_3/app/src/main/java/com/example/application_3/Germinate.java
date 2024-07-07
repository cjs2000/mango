package com.example.application_3;

import android.content.ContentProviderOperation;
import android.content.Intent;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.nfc.Tag;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import com.bumptech.glide.Glide;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

import io.grpc.Context;
import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONException;




public class Germinate extends AppCompatActivity {

    private String Lan="http://192.168.137.1:8000/";

    private String Ngrok="https://frankly-content-platypus.ngrok-free.app/";

    private OkHttpClient client = new OkHttpClient();
    private Context context;
    private Bitmap selectedImageBitmap; // 用于存储从相册选择的图片
    private Uri selectedImageUri; // 用于存储选定图片的URI
    private Uri imageUri;        // 用于存储拍摄图片的URI

    private ImageView loadView;
    File outputImage;
    private static final int REQUEST_SELECT_IMAGE = 1; // 请求代码，用于标识选择图片的操作
    private static final int REQUEST_CAMERA_IMAGE = 2;      // 相机请求代码

    Button recButton;
    Button SelButton;
    Button hisButton;
    Button cameraButton;

    TextView GalleryText;
    TextView CameraText;
    TextView RecognizeText;
    TextView HistoryText;

    String ImageName;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.germinate);

        this.context = context;

        loadView = findViewById(R.id.imageView);

        // 加载 GIF 动图
        Glide.with(this)
                .load(R.drawable.load)
                .into(loadView);
        loadView.setVisibility(View.GONE);


        Bitmap initImage;
        // 获取 Resources 对象
        Resources resources = getResources();
        // 获取 pic.jpg 图片的 Drawable 对象
        Drawable drawable = resources.getDrawable(R.drawable.pic);
        // 将 Drawable 对象转换为 Bitmap 对象
        Bitmap bitmap = ((BitmapDrawable) drawable).getBitmap();
        // 将 Bitmap 对象设置给 initImage
        initImage = bitmap;
        ImageView imageView = findViewById(R.id.imageView_g);
        imageView.setImageBitmap(initImage);

        //获取按钮
        recButton = (Button) findViewById(R.id.recognizeButton_g);
        SelButton = (Button) findViewById(R.id.chooseButton_g);         // 按钮ID
        GalleryText =(TextView) findViewById(R.id.gallery);
        CameraText =(TextView) findViewById(R.id.Camera);
        RecognizeText = (TextView) findViewById(R.id.Send);
        HistoryText = (TextView) findViewById(R.id.History);

        //打开图库，选择图片，并记录下被选图片的地址
        SelButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // 创建一个Intent，启动图库应用
                Intent intent = new Intent(Intent.ACTION_PICK);
                intent.setType("image/*"); // 限制只选择图片
                startActivityForResult(intent, REQUEST_SELECT_IMAGE);
            }
        });

        //将选中的图片发送给后端，后端端口：http://192.168.137.1:8000/process/load/
        recButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(selectedImageBitmap==null) {
                    Toast.makeText(Germinate.this, "You did not select a picture or take a picture.", Toast.LENGTH_SHORT).show();
                    return;
                }
                uploadImage(selectedImageBitmap);
            }
        });

        // 发送Get请求，向Django后端发送请求：http://192.168.137.1:8000/process/load/
        hisButton = (Button) findViewById(R.id.hisButton_g);
        hisButton .setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Germinate.this, History.class);
                startActivity(intent);
            }
        });


        cameraButton = (Button) findViewById(R.id.cameraButton_g);
        //点击相机按钮，将相机获取的画面投射在ImageView上
        cameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // 指定拍照
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                // 指定存储路径
                imageUri = FileProvider.getUriForFile(Germinate.this,
                        "com.example.application_3.fileprovider",
                        createImageFile());
                intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
                // 拍照返回图片
                startActivityForResult(intent, REQUEST_CAMERA_IMAGE);
            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_SELECT_IMAGE) {

                if (data != null) {
                    // 获取用户选择的图片的URI
                    selectedImageUri = data.getData();
                    if (selectedImageUri != null) {
                        // 获取图片数据
                        selectedImageBitmap = loadImageFromUri(selectedImageUri);
                        // 使用Picasso加载并显示图片到ImageView
                        ImageView imageView = findViewById(R.id.imageView_g);
                        imageView.setImageBitmap(selectedImageBitmap);
                        if (selectedImageBitmap != null) {
                            Toast.makeText(Germinate.this, "Image selected", Toast.LENGTH_SHORT).show();

                        }
                    }
                }
            } else if (requestCode == REQUEST_CAMERA_IMAGE) {
                if (imageUri != null) {
                    selectedImageBitmap = loadImageFromUri(imageUri);
                    // 将Bitmap显示在ImageView上
                    ImageView imageView = findViewById(R.id.imageView_g);
                    imageView.setImageBitmap(selectedImageBitmap);
                    if (selectedImageBitmap == null) {
                        Toast.makeText(Germinate.this, "拍摄图片未找到，uri为:" + imageUri.toString(), Toast.LENGTH_SHORT).show();
                    }
                }
            }
        }
    }




    private Bitmap loadImageFromUri(Uri uri) {
        try {
            InputStream inputStream = getContentResolver().openInputStream(uri);
            return BitmapFactory.decodeStream(inputStream);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    // 创建图片文件
    private File createImageFile() {
        // 获取文件名
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + ".jpg";
        // 创建文件
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File imageFile = new File(storageDir, imageFileName);
        return imageFile;
    }

    private File createImageFile_1() {
        // 获取文件名
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_1" + timeStamp + ".jpg";
        // 创建文件
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File imageFile = new File(storageDir, imageFileName);
        return imageFile;
    }

    private File createImageFile_2() {
        // 获取文件名
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_2" + timeStamp + ".jpg";

        // 创建文件
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File imageFile = new File(storageDir, imageFileName);

        return imageFile;
    }

    //将Bitmap文件上传至服务器，服务器后端端口：https://frankly-content-platypus.ngrok-free.app/process/load/
    private void uploadImage(Bitmap image) {

        if (image == null) {
            Toast.makeText(Germinate.this, "Image not selected", Toast.LENGTH_SHORT).show();
        }
        cameraButton.setVisibility(View.GONE);
        hisButton.setVisibility(View.GONE);
        recButton.setVisibility(View.GONE);
        SelButton.setVisibility(View.GONE);
        GalleryText.setVisibility(View.GONE);
        CameraText.setVisibility(View.GONE);
        RecognizeText.setVisibility(View.GONE);
        HistoryText.setVisibility(View.GONE);

        ImageView imageView = findViewById(R.id.imageView_g);
        imageView.setVisibility(View.GONE);
        loadView.setVisibility(View.VISIBLE);

        // 创建OkHttpClient
        OkHttpClient client = new OkHttpClient.Builder()
                .connectTimeout(60, TimeUnit.SECONDS) // 设置连接超时时间
                .readTimeout(60, TimeUnit.SECONDS)    // 设置读取超时时间
                .writeTimeout(60, TimeUnit.SECONDS)   // 设置写入超时时间
                .build();

        // 将Bitmap转换为字节数组
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        image.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
        byte[] imageBytes = byteArrayOutputStream.toByteArray();

        // 构建Multipart请求体

        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", "image.jpg", RequestBody.create(MediaType.parse("image/jpeg"), imageBytes))
                .build();




        // 创建POST请求
        Request request = new Request.Builder()
                //.url(Ngrok+"process/germinate/") // 服务器地址
                .url(Lan+"process/mango/") // 服务器地址
                .post(requestBody)
                .build();

        // 记录请求发送时间
        final long startTime = System.currentTimeMillis();
        // 发送请求
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                // 请求失败时的处理
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(Germinate.this, "上传失败：" + e.toString(), Toast.LENGTH_SHORT).show();
                    }
                });
            }



            @Override
            public void onResponse(Call call, Response response) throws IOException {
                // 请求成功时的处理
                final byte[] imageBytes = response.body().bytes(); // 获取后端的字节数组


                // 记录响应接收时间
                final long endTime = System.currentTimeMillis();
                final long duration = endTime - startTime;
                final String durationStr = String.valueOf(duration);

                //Toast.makeText(Germinate.this, "耗时：" + durationStr, Toast.LENGTH_SHORT).show();

                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        String imageDataString = new String(imageBytes, StandardCharsets.UTF_8);

                        JSONObject responseJson = null;
                        try {
                            responseJson = new JSONObject(imageDataString);
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }

                        String id;
                        String confidence;
                        String runTime;


                        // 获取数据
                        try {
                            id = responseJson.getString("id");
                            confidence = responseJson.getString("confidence");
                            runTime = responseJson.getString("runTime");

                        } catch (JSONException e) {
                            throw new RuntimeException(e);
                        }

                        String reslut[]={id,confidence,durationStr};

                        Intent intent = new Intent(Germinate.this, Result.class);
                        intent.putExtra("reslut", reslut);
                        startActivity(intent);
                        cameraButton.setVisibility(View.VISIBLE);
                        hisButton.setVisibility(View.VISIBLE);
                        recButton.setVisibility(View.VISIBLE);
                        SelButton.setVisibility(View.VISIBLE);
                        ImageView imageView = findViewById(R.id.imageView_g);
                        imageView.setVisibility(View.VISIBLE);

                        GalleryText.setVisibility(View.VISIBLE);
                        CameraText.setVisibility(View.VISIBLE);
                        RecognizeText.setVisibility(View.VISIBLE);
                        HistoryText.setVisibility(View.VISIBLE);
                        loadView.setVisibility(View.GONE);
                    }
                });
            }



        });
    }

}
