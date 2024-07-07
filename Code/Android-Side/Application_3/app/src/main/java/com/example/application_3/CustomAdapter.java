package com.example.application_3;

import android.content.Context;
import android.graphics.Bitmap;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.List;

public class CustomAdapter extends ArrayAdapter<String> {

    private int resourceId;
    private List<Bitmap> imageList;

    public CustomAdapter(Context context, int resource, List<String> objects, List<Bitmap> images) {
        super(context, resource, objects);
        this.resourceId = resource;
        this.imageList = images;
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        String text = getItem(position);
        Bitmap image = imageList.get(position);
        View view;
        ViewHolder viewHolder;
        if (convertView == null) {
            view = LayoutInflater.from(getContext()).inflate(resourceId, parent, false);
            viewHolder = new ViewHolder();
            viewHolder.textView = view.findViewById(R.id.textView);
            viewHolder.imageView = view.findViewById(R.id.imageView);
            view.setTag(viewHolder);
        } else {
            view = convertView;
            viewHolder = (ViewHolder) view.getTag();
        }
        viewHolder.textView.setText(text);
        viewHolder.imageView.setImageBitmap(image);
        return view;
    }

    class ViewHolder {
        TextView textView;
        ImageView imageView;
    }
}
