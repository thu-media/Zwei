#include "PEOPLE.hh"
#include <stdio.h>
#define MTU_SIZE 1500
PEOPLE::PEOPLE()
{
}

int PEOPLE::peopleSendMessage(char *data, int length)
{
	char *p = data;
	int len = length;
	FILE* file = fopen("test.log","a+");
	while (len > 0)
	{
		int size = len > MTU_SIZE ? MTU_SIZE : len;
		SDLNet_TCP_Send(this->sock, p, size);
		fprintf(file,"%d,%d\n",SDL_GetTicks(),1);
		len -= size;
		p += size;
	}
	fclose(file);
	return length;
	//return SDLNet_TCP_Send(this->sock, data, length);
	//	return 0;
}

PEOPLE::~PEOPLE()
{
}
