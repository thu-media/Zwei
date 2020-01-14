#pragma once

#include <stdint.h>
#include <stdio.h>
extern "C" {
#ifndef INT64_C
#define INT64_C(c) (c##LL)
#define UINT64_C(c) (c##ULL)
#endif
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavutil/avutil.h"
#include "libavutil/opt.h" // for av_opt_set
#include <x264.h>
#include <x264_config.h>
}

#define RBR_MAX_NUM_REF 2
#define RBR_MIN_ENCODING_RATE 96
#define RBR_MAX_ENCODING_RATE 3000
#define RBR_READ_UNTIL 288
#define RBR_BUFFER_SIZE_COEFF 5
class mythX264Encoder
{
  public:
    static void RGB2yuv(int width, int height, int stride, const void *src, void **dst);
    typedef void(responseHandler)(void *myth, char *pdata, int plength);
    mythX264Encoder(void *phwnd, int width, int height);
    ~mythX264Encoder(void);
    bool Init();
    void Cleanup();
    void ProcessFrame(unsigned char **src, int *srclinesize, responseHandler *response);

    void Reconf(int bitrate_value, int var_value);
    static mythX264Encoder *CreateNew(void *hwnd, int width, int height);
    void *hwnd;

  protected:
    int mwidth;
    int mheight;

  private:
    x264_t *encoder;
    x264_nal_t *nal;
    x264_param_t param;
    x264_picture_t pic_in, pic_out;

    int i_frame_size;
    int i_nal;
    unsigned u_size[3];
    //unsigned u_luma_size;
    //unsigned u_chroma_size;

    int i_frame_index;  // index of the frame in the intra-period
    int i_last_idr = 0; // i_enc_frame number of the last IDR frame
    int i_trim = 1;     // FR reduction factor, default: no reduction

    const double d_ipr = 16.0 / 15; // intra-period duration in seconds
    double d_fr;                    // encoding frame rate
    int i_ipr = 32;
    uint64_t ul_raw_ts = 0;
    double d_interval = 0;
};
