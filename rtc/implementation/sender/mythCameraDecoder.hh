#pragma once
#include "mythconfig.hh"
#include "mythVirtualDecoder.hh"
#include "mythCamera.hh"
#include "mythX264Encoder.hh"
#include "mythFFmpegEncoder.hh"
class mythCameraDecoder :
	public mythVirtualDecoder
{
public:
	static void staticresponse(void *myth, char* pdata, int plength);
	static mythCameraDecoder* CreateNew(){
		return new mythCameraDecoder();
	}
	void response(char* pdata, int plength);
	void start();
	void stop();
	void reconfig(int value, int var);
	~mythCameraDecoder();
protected:
	int decodethread();
	mythCameraDecoder();
	static int decodethreadstatic(void* data);
	mythFFmpegEncoder* encoder;
	SDL_Thread* startthread;
	SDL_mutex* startmutex;
	mythCamera* camera;
};

