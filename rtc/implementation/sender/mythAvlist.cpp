#include "mythAvlist.hh"
#include <stdio.h>
//#include <memory.h>
#ifdef ANDROID
#include <android/log.h>
#define LOG_TAG "org.app.sdl"

#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#else
#define LOGE printf
#endif
#define AVFRAMECOUNT 25
#define AVTOTALBUFFERCOUNT 5 * 1024 * 1024//5M
mythAvlist::mythAvlist(void)
{
	listcount = 0;
	this->startread = false;
	this->mutex = SDL_CreateMutex();
	this->totalptr = 0;
	InitalList();
}
int mythAvlist::InitalList(void){
	//inital list
	totalbuffer = new unsigned char[AVTOTALBUFFERCOUNT];
	ListPacket = new PacketQueue[AVFRAMECOUNT];
	for(int i = 0;i < AVFRAMECOUNT;i++){
		//ListPacket[i].YY = NULL;
		//ListPacket[i].UU = NULL;
		//ListPacket[i].VV = NULL;
		ListPacket[i].h264Packet = NULL;
	}
	listwrite = 0;
	listread = 0;
	return 0;
}
mythAvlist::mythAvlist(int BufferSize)
{
	listcount = 0;
	mutex = SDL_CreateMutex();
	totalbuffer = (unsigned char*)SDL_malloc(BufferSize * 1024 * 1024);
	totalptr = 0;
	InitalList();
}
mythAvlist *mythAvlist::CreateNew(int BufferSize){
	if(BufferSize == -1)
		return new mythAvlist();
	else
		return new mythAvlist(BufferSize);
}
mythAvlist::~mythAvlist(void)
{
	free();
}
PacketQueue *mythAvlist::get(int freePacket){
	SDL_LockMutex(this->mutex);
	PacketQueue *tmp;
	if (this->listwrite - this->listread == 1 ||
		this->listwrite - this->listread == 0 ||
		(this->listwrite == 0 && this->listread == AVFRAMECOUNT)){
		tmp = NULL;
	}else{
		tmp = &this->ListPacket[this->listread];
		if(tmp->h264Packet == NULL){
			tmp = NULL;
		}else{
			if(freePacket == 0){
				if(listwrite - listread > 10){
					//LOGE("skip frames");
					//LOGE(" read = %d,write = %d,minus = %d\n",listread,listwrite,listwrite - listread);
					listread += 9;
				}else
					listread++;
			}
		}
	}
	listread %= AVFRAMECOUNT;
	SDL_UnlockMutex(this->mutex);
	return tmp;
}
unsigned char* mythAvlist::putcore(unsigned char* data,unsigned int datasize){
	if(totalptr + datasize >= AVTOTALBUFFERCOUNT)totalptr = 0;
	SDL_memcpy(totalbuffer + totalptr, data, datasize);
	totalptr += datasize;
	//printf("totalptr = %d\n",totalptr);
	return (totalbuffer + totalptr - datasize);
}
/*
int mythAvlist::put(unsigned char** dataline,unsigned int *datasize,unsigned int width,unsigned int height){
	SDL_LockMutex(this->mutex);
	if(listwrite >= AVFRAMECOUNT)listwrite = 0;
	PacketQueue *tmp = &ListPacket[listwrite];
	tmp->h264Packet = NULL;
	tmp->width = width;
	tmp->height = height;

	tmp->YY = (unsigned char*)putcore(dataline[0],datasize[0] * height);
	tmp->Ydatasize = datasize[0];
	
	tmp->UU = (unsigned char*)this->putcore(dataline[1], datasize[1] * height / 2);
	tmp->Udatasize = datasize[1];
	
	tmp->VV = (unsigned char*)this->putcore(dataline[2], datasize[2] * height / 2);
	tmp->Vdatasize = datasize[2];

	listwrite++;
	//LOGE("YUVlistcount=%d\n",listwrite);
	SDL_UnlockMutex(this->mutex);
	return 0;
}
*/
int mythAvlist::release(PacketQueue *pack)
{
	return 0;
}
int mythAvlist::put(unsigned char* data,unsigned int length){	
	if (!mutex){ return 1; }
	SDL_LockMutex(this->mutex);
	if(listwrite >= AVFRAMECOUNT)listwrite = 0;
	PacketQueue *tmp = &ListPacket[listwrite];

	//tmp->YY = NULL;
	//tmp->UU = NULL;
	//tmp->VV = NULL;

	tmp->h264PacketLength = length;
	tmp->h264Packet = putcore(data, length);
	listwrite++;
	//LOGE("H264listcount=%d\n",listwrite);
	SDL_UnlockMutex(this->mutex);
	return 0;
}
int mythAvlist::free(){
	if (mutex)
		SDL_DestroyMutex(mutex);
	mutex = NULL;
	if (ListPacket)
		delete [] ListPacket;
	ListPacket = NULL;
	if (totalbuffer)
		delete [] totalbuffer;
	totalbuffer = NULL;
	return 0;
}
