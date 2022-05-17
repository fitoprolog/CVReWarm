package com.test.cvrewarm;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Camera;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.util.Log;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;

import org.opencv.core.MatOfInt;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import android.content.Context;
import java.io.File;
import java.lang.reflect.Method;
import java.util.List;
import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import java.util.Scanner;
import java.text.DecimalFormat;
import java.util.Random;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import java.util.Collections;
import android.content.pm.ActivityInfo;
import org.tensorflow.lite.Interpreter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import android.content.res.AssetFileDescriptor;
import android.app.Activity;
import java.io.IOException;

public class MainActivity extends AppCompatActivity  implements /*OnTouchListener,*/ CvCameraViewListener2 {
    private static final String  TAG              = "MainActivity";

    private boolean              mIsColorSelected = false;
    private Mat                  mRgba;
    private Scalar               mBlobColorRgba;
    private Scalar               mBlobColorHsv;
    private ColorBlobDetector    mDetector;
    private Mat                  mSpectrum;
    private Size                 SPECTRUM_SIZE;
    private Scalar               CONTOUR_COLOR;
    private Net net;
    private static List<String> classNames;
    private static List<Scalar> colors=new ArrayList<>();
    private CameraBridgeViewBase mOpenCvCameraView;
    private static final int MY_CAMERA_REQUEST_CODE = 100;
    protected ByteBuffer imgData = null;
    private Interpreter tflite;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    //mOpenCvCameraView.setOnTouchListener(MainActivity.this);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("output.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);
        classNames = readLabels("labels.txt", this);
        for(int i=0; i<classNames.size(); i++)
            colors.add(randomColor());
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.color_blob_detection_activity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraPermissionGranted();

        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, MY_CAMERA_REQUEST_CODE);
        }

    }
    protected void setDisplayOrientation(Camera camera, int angle){
        Method downPolymorphic;
        try
        {
            downPolymorphic = camera.getClass().getMethod("setDisplayOrientation", new Class[] { int.class });
            if (downPolymorphic != null)
                downPolymorphic.invoke(camera, new Object[] { angle });
        }
        catch (Exception e1)
        {
            e1.printStackTrace();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MY_CAMERA_REQUEST_CODE) {

            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show();
            } else {
                Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallback);

        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }
    private static String getAssetsFile(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mDetector = new ColorBlobDetector();
        mSpectrum = new Mat();
        mBlobColorRgba = new Scalar(255);
        mBlobColorHsv = new Scalar(255);
        SPECTRUM_SIZE = new Size(200, 64);
        CONTOUR_COLOR = new Scalar(255,0,0,255);
        File file = new File(getAssetsFile("yolov3-tiny_training.cfg",this));
        imgData = ByteBuffer.allocateDirect( 128 * 128 * 3 * 4);
        imgData.order(ByteOrder.nativeOrder());
        if(file.exists())
            Toast.makeText(this,"YUSH" , Toast.LENGTH_LONG).show();
        else
            Toast.makeText(this,"NO" , Toast.LENGTH_LONG).show();
        net = Dnn.readNetFromDarknet(getAssetsFile("yolov3-tiny_training.cfg",this),
                getAssetsFile( "yolov3-tiny_training_last.weights",this));
        if ( net.empty() ) {
            Log.d(TAG, "ERROR LOADING WEIGHTS");
            //Toast.makeText(this,"ERROR LOADING WEIGHTS" , Toast.LENGTH_LONG).show();
        }else
        {
            Log.d(TAG, "WEIGHTS LOADED CORRECTLY");
            //Toast.makeText(this,"WEIGHTS LOADED CORRECTLY" , Toast.LENGTH_LONG).show();
        }
        try{
          tflite = new Interpreter(loadModelFile(this));
          tflite.allocateTensors();
          Log.d(TAG, "PASSED TFLITE PART :D");
        }catch(IOException e) {
            Log.e("FAILTFLITE",e.toString());
        }
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    /*public boolean onTouch(View v, MotionEvent event) {
        rett
    }*/
    private List<String> readLabels (String file, Context context)
    {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream;
        List<String> labelsArray = new ArrayList<>();
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            Scanner fileScanner = new Scanner(new File(outFile.getAbsolutePath())).useDelimiter("\n");
            String label;
            while (fileScanner.hasNext()) {
                label = fileScanner.next();
                labelsArray.add(label);
            }
            fileScanner.close();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to read labels!");
        }
        return labelsArray;
    }
    private void convertMattoTfLiteInput(Mat mat)
    {
        imgData.rewind();
        int pixel = 0;
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                imgData.putFloat((float)mat.get(i,j)[0]);
                imgData.putFloat((float)mat.get(i,j)[1]);
                imgData.putFloat((float)mat.get(i,j)[2]);
            }
        }
    }
    private void ToMat(float [][][] d, Mat mat){
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                mat.put(i,j,d[i][j]);
            }
        }
    }
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGBA2RGB);
        Mat resized = new Mat(128, 128, CvType.CV_8UC3);
        Mat outMat  = new Mat( 128,128,CvType.CV_32FC(3));
        float [][][][] out = new float [1][128][128][3];
        Imgproc.resize(mRgba,resized,new Size(128,128));
        convertMattoTfLiteInput(resized);
        tflite.run(imgData, out);
        ToMat(out[0],outMat);
        outMat.convertTo(resized,CvType.CV_8UC3);
        Imgproc.resize(resized,mRgba,new Size(mRgba.width(),mRgba.height()));
        /*Mat gray = new Mat(mRgba.height(),mRgba.width(),CvType.CV_8   UC1);
        Imgproc.cvtColor(mRgba,gray,Imgproc.COLOR_RGBA2GRAY);Pixel_3a_API_30_x86
        Imgproc.threshold(gray,gray,10,255,Imgproc.THRESH_BINARY);  
        Imgproc.cvtColor(gray,mRgba,Imgproc.COLOR_GRAY2BGRA);*/
        return mRgba;
    }

    private Scalar converScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0, 0));
    }

    /*@Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }*/

    private Scalar randomColor() {
        Random random = new Random();
        int r = random.nextInt(255);
        int g = random.nextInt(255);
        int b = random.nextInt(255);
        return new Scalar(r,g,b);
    }

    private static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();

        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
        return names;
    }

}