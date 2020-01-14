#pragma once
#ifdef WIN32
#include <WinSock2.h>
#pragma comment(lib, "ws2_32")
typedef int socklen_t;
#else
#include <wchar.h>
#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#endif
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include "cJSON.h"
#include "mythCameraDecoder.hh"
#define HOSTIP "127.0.0.1"
#define HOSTPORT 23001
class VirtualP2PServer
{
  public:
	static VirtualP2PServer *CreateNew()
	{
		return new VirtualP2PServer();
	}
	VirtualP2PServer();
	~VirtualP2PServer();
	static void *StartThreadStatic(void *data)
	{
		if (data)
		{
			VirtualP2PServer *tmp = (VirtualP2PServer *)data;
			tmp->StartThread();
		}
		return NULL;
	}

	int Start();
	int Stop();

  private:
	unsigned long mythTickCount()
	{
#ifdef WIN32
		return GetTickCount();
#else
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
#endif
	}
	int InitSock();

  protected:
	void SendStream(mythCameraDecoder *decoder, sockaddr_in *sender);
	void udpsend(sockaddr_in *addr, char *data, int len);
	void udpsend(char *ip, int port, cJSON *msg);
	void udpsend(sockaddr_in *addr, cJSON *msg);
	void hostsend(cJSON *msg);

	int StartThread();
#ifdef WIN32
	SOCKET PrimaryUDP;
#else
	int PrimaryUDP;
#endif
	bool mrunning;
	int recv_within_time(int fd, char *buf, size_t buf_n, struct sockaddr *addr, socklen_t *len, unsigned int sec, unsigned usec);
};
