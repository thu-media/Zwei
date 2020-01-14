#pragma once

#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include <iostream>

#ifdef ANDROID
#define MYTHCAMERAPOS CV_CAP_ANDROID_BACK
#else
#define MYTHCAMERAPOS CV_CAP_ANY
#endif
using namespace std;
class mythCamera
{
public:
	static mythCamera* mmythCamera;
	static mythCamera* GetInstance(){
		if (!mmythCamera)
			return CreateNew();
		else
			return mmythCamera;
	}
	static mythCamera* CreateNew(){
		return new mythCamera();
	}
	~mythCamera();
	void CloseCamera();
	int Capture(int* width, int *height, int* stride, void** imageSource);
protected:
	mythCamera();
private:
	IplImage* pFrame;
	CvCapture* pCapture;
};

