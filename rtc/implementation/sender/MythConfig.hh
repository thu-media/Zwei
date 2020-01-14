#pragma once
#include <stdio.h>
#ifdef ANDROID
#	include "SDL.h"
#	include "SDL_net.h"
#else
#	include "SDL2/SDL.h"
#	include "SDL2/SDL_net.h"
#endif
#define mythcmp(A) strcmp(input,A) == 0
#define streamserverport 5834

//#define MYTH_CONFIG_SENDMESSAGE_SLOW
#define MYTH_CONFIG_SENDMESSAGE_FAST
#define MYTH_INFORMATIONINI_FILE "mythconfig.ini"