#pragma once
#include "MythConfig.hh"
class PEOPLE
{
public:
	int peopleSendMessage(char* data, int length);
	static PEOPLE* CreateNew(){
		return new PEOPLE();
	}
	PEOPLE();
	~PEOPLE();
	int active;
	TCPsocket sock;
	IPaddress peer;
	void* addtionaldata;
};

