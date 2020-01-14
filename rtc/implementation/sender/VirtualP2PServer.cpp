#include "VirtualP2PServer.hh"
#include "mythCameraDecoder.hh"

#define BUFFERMAX 1400

VirtualP2PServer::VirtualP2PServer()
{
	InitSock();
	mrunning = false;
}

VirtualP2PServer::~VirtualP2PServer()
{
}

int VirtualP2PServer::InitSock()
{
#ifdef WIN32
	WSADATA wsaData;

	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
	{
		printf("Windows sockets 2.2 startup");
		return 1;
	}
	else
	{
		printf("init winsock32 success\n");
	}
#endif
	return 0;
}

int VirtualP2PServer::Stop()
{
	mrunning = false;
	return 0;
}

#define RECV_LOOP_COUNT 1
int VirtualP2PServer::recv_within_time(int fd, char *buf, size_t buf_n, struct sockaddr *addr, socklen_t *len, unsigned int sec, unsigned usec)
{
#if 1
	struct timeval tv;
	fd_set readfds;
	int i = 0;
	unsigned int n = 0;
	for (i = 0; i < RECV_LOOP_COUNT; i++)
	{
		FD_ZERO(&readfds);
		FD_SET(fd, &readfds);
		tv.tv_sec = sec;
		tv.tv_usec = usec;
		select(fd + 1, &readfds, NULL, NULL, &tv);
		if (FD_ISSET(fd, &readfds))
		{
			if ((n = recvfrom(fd, buf, buf_n, 0, addr, len)) >= 0)
			{
				return n;
			}
		}
	}
	return -1;
#else
	return recvfrom(fd, buf, buf_n, 0, addr, len);
#endif
}

int VirtualP2PServer::Start()
{
	StartThreadStatic(this);
	//pthread_create(&pid, NULL, StartThreadStatic, this);
	return 0;
}

void VirtualP2PServer::udpsend(sockaddr_in *addr, cJSON *msg)
{
	char *msgstr = cJSON_PrintUnformatted(msg);
	sendto(PrimaryUDP, msgstr, strlen(msgstr), 0, (sockaddr *)addr, sizeof(sockaddr));
}

void VirtualP2PServer::udpsend(sockaddr_in *addr, char *data, int len)
{
	char buffer[BUFFERMAX] = {0};
	unsigned long tick = mythTickCount();
	SDL_memcpy(buffer, &tick, sizeof(tick));
	SDL_memcpy(buffer + sizeof(tick), data, len);
	sendto(PrimaryUDP, buffer, len + sizeof(tick), 0, (sockaddr *)addr, sizeof(sockaddr));
}

void VirtualP2PServer::udpsend(char *ip, int port, cJSON *msg)
{
	struct sockaddr_in addr = {0};
	addr.sin_family = AF_INET;
	addr.sin_port = htons(port);
	addr.sin_addr.s_addr = inet_addr(ip);
	char *msgstr = cJSON_PrintUnformatted(msg);
	sendto(PrimaryUDP, msgstr, strlen(msgstr), 0, (sockaddr *)&addr, sizeof(sockaddr));
}

void VirtualP2PServer::hostsend(cJSON *msg)
{
	udpsend(HOSTIP, HOSTPORT, msg);
}

void VirtualP2PServer::SendStream(mythCameraDecoder *decoder, sockaddr_in *sender)
{
	PacketQueue *tmp = NULL;
	for (;;)
	{
		tmp = decoder->get();
		if (!tmp)
		{
			break;
		}
		else
		{
			char *_data = (char *)tmp->h264Packet;
			int _index = 0;
			int _len = tmp->h264PacketLength;
			int _size = BUFFERMAX - 8;
			do
			{
				int _send_len = _len > _size ? _size : _len;
				udpsend(sender, _data + _index, _send_len);
				_index += _send_len;
				_len -= _send_len;
			} while (_len > 0);
		}
	}
}

int VirtualP2PServer::StartThread()
{
	PrimaryUDP = socket(AF_INET, SOCK_DGRAM, 0);
	if (PrimaryUDP < 0)
	{
		printf("create socket error");
		return 1;
	}
	sockaddr_in local;
	local.sin_family = AF_INET;
	local.sin_port = htons(23583);
	local.sin_addr.s_addr = INADDR_ANY;
	int nResult = bind(PrimaryUDP, (sockaddr *)&local, sizeof(sockaddr));
	//mrunning = true;
	sockaddr_in sender = {0};
	int dwSender = sizeof(sender);
	char recvbuf[1501] = {0};
	printf("Start Loop! Port:%d\n", 23583);
	unsigned long tick = mythTickCount();
	unsigned long basictick = tick;
	mythCameraDecoder *decoder = mythCameraDecoder::CreateNew();
	decoder->start();
	while (1)
	{
		memset(recvbuf, 0, 1500);
		int ret = recv_within_time(PrimaryUDP, (char *)&recvbuf, 1500, (sockaddr *)&sender, (socklen_t *)&dwSender, 0, 10);
		if (ret > 0)
		{
			//printf("ret=%d\n", ret);
			if (recvbuf[0] == '{' && recvbuf[ret - 1] == '}')
			{
				//printf("%s\n", recvbuf);
				cJSON *root = cJSON_Parse(recvbuf);
				if (root)
				{
					//cJSON* ret;
					cJSON *req_baseline = cJSON_GetObjectItem(root, "baseline");
					cJSON *req_var = cJSON_GetObjectItem(root, "variable");
					if (req_baseline && req_var)
					{
						int _baseline = req_baseline->valueint;
						int _variable = req_var->valueint;
						printf("%d,%d\n", _baseline, _variable);
						decoder->reconfig(_baseline, _variable);
					}
				}
				else
				{
					//maybe in p2p model
					printf("json parse error!\n");
				}
			}
		}
		//if (sender.sin_port == 0)
		{
			SendStream(decoder, &sender);
		}
	}
	return 0;
}
