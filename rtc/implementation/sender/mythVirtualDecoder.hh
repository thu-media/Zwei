#pragma once
#include <stdio.h>
#include "mythAvlist.hh"

class mythVirtualDecoder :
	public mythAvlist
{
public:
	static mythVirtualDecoder* CreateNew(void);
	virtual void start();
	virtual void stop();
	virtual ~mythVirtualDecoder(void);
protected:
	mythVirtualDecoder(void);
	int flag;
};

