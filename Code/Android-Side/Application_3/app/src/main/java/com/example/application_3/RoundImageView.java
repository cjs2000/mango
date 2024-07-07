package com.example.application_3;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.util.AttributeSet;
import androidx.appcompat.widget.AppCompatImageView;

public class RoundImageView extends AppCompatImageView {
    private final Path path = new Path();
    private final RectF rect = new RectF();
    private float cornerRadius = 70.0f; // 设置圆角半径，默认是20dp

    public RoundImageView(Context context) {
        super(context);
    }

    public RoundImageView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public RoundImageView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        if (getDrawable() != null) {
            int saveCount = canvas.saveLayer(0, 0, getWidth(), getHeight(), null, Canvas.ALL_SAVE_FLAG);

            path.reset();
            rect.set(0, 0, getWidth(), getHeight());
            path.addRoundRect(rect, cornerRadius, cornerRadius, Path.Direction.CW);
            canvas.clipPath(path);

            super.onDraw(canvas);

            canvas.restoreToCount(saveCount);
        } else {
            super.onDraw(canvas);
        }
    }
    public void setCornerRadius(float radius) {
        cornerRadius = radius;
        invalidate();
    }
}
