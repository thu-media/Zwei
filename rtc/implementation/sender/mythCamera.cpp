#include "mythCamera.hh"
#include "MythConfig.hh"
mythCamera* mythCamera::mmythCamera = NULL;
mythCamera::mythCamera()
{
	pCapture = cvCreateCameraCapture(MYTHCAMERAPOS);
	pFrame = NULL;
	if (pCapture)
		cout << "start OK!" << endl;
	else
		cout << "start failed!" << endl;
	this->mmythCamera = this;
}
void mythCamera::CloseCamera(){
	cvReleaseImage(&pFrame);
}
int mythCamera::Capture(int* width, int *height, int* stride,void** imageSource){
	//int t = SDL_GetTicks();
	pFrame = cvQueryFrame(pCapture);
	//printf("capture time: %dms\n", SDL_GetTicks() - t);
	if (pFrame){
		if(width)*width = pFrame->width;
		if(height)*height = pFrame->height;
		if (stride)*stride = pFrame->widthStep;
		if (imageSource)*imageSource = pFrame->imageData;
        //puts(pFrame->channelSeq);
		return 0;
	}
	else{
		return 1;
	}
}

mythCamera::~mythCamera()
{
}
