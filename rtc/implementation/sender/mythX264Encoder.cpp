#include "mythX264Encoder.hh"
#include <stdio.h>
mythX264Encoder::mythX264Encoder(void *phwnd, int width, int height)
{

    mwidth = width;
    mheight = height;
    hwnd = phwnd;
    Init();
}

void mythX264Encoder::RGB2yuv(int width, int height, int stride, const void *src, void **dst)
{
    struct SwsContext *img_convert_ctx = sws_getContext(
        width, height, AV_PIX_FMT_RGB24,
        width, height, AV_PIX_FMT_YUV420P,
        SWS_FAST_BILINEAR, NULL, NULL, NULL);
    uint8_t *rgb_src[3] = {(uint8_t *)src, NULL, NULL};

    int srcwidth[] = {stride, 0, 0};
    int dstwidth[] = {width, width / 2, width / 2};
    if (img_convert_ctx)
    {
        sws_scale(img_convert_ctx, (const uint8_t *const *)rgb_src, srcwidth, 0, height,
                  (uint8_t *const *)dst, dstwidth);
    }
    sws_freeContext(img_convert_ctx);
    return;
}

mythX264Encoder *mythX264Encoder::CreateNew(void *phwnd, int width, int height)
{
    return new mythX264Encoder(phwnd, width, height);
}

bool mythX264Encoder::Init()
{
    nal = NULL;
    encoder = NULL;
    //x264_param_default_preset(&param, "veryfast", "zerolatency");

    x264_param_default(&param);
    param.i_width = mwidth;
    param.i_height = mheight;
    param.i_log_level = X264_LOG_DEBUG;
    param.i_threads = X264_SYNC_LOOKAHEAD_AUTO;
    param.i_frame_total = 0;
    param.i_keyint_max = 10;
    param.i_bframe = 0;
    param.b_open_gop = 0;
    param.i_bframe_pyramid = 0;
    param.rc.i_qp_constant = 0;
    param.rc.i_qp_max = 0;
    param.rc.i_qp_min = 0;
    param.i_bframe_adaptive = X264_B_ADAPT_TRELLIS;
    param.i_fps_den = 1;
    param.i_fps_num = 5;
    param.i_timebase_den = param.i_fps_num;
    param.i_timebase_num = param.i_fps_den;
    param.i_csp = X264_CSP_I420;
    param.rc.i_rc_method = X264_RC_ABR;
    param.rc.i_bitrate = 120;
    param.rc.i_vbv_max_bitrate = 120;

    u_size[0] = param.i_width * param.i_height; // set for each frame in case SR changes
    u_size[1] = u_size[0] / 4;                  // set for each frame in case SR changes
    u_size[2] = u_size[0] / 4;                  // set for each frame in case SR changes

    /* Apply profile restrictions. */
    x264_param_apply_profile(&param, "high");
    encoder = x264_encoder_open(&param);

    x264_picture_init(&pic_out);
    x264_picture_init(&pic_in);
    x264_picture_alloc(&pic_in, param.i_csp, param.i_width, param.i_height);
    return true;
}

mythX264Encoder::~mythX264Encoder(void)
{
    Cleanup();
}

void mythX264Encoder::Reconf(int bitrate_value, int var_value)
{
    double _newbitrate_ = bitrate_value / 1000.0;
    _newbitrate_ = _newbitrate_ <= RBR_MIN_ENCODING_RATE ? RBR_MIN_ENCODING_RATE : (_newbitrate_ >= RBR_MAX_ENCODING_RATE ? RBR_MAX_ENCODING_RATE : _newbitrate_);
    printf("[RBR] target=%.2f\n", _newbitrate_);
    x264_encoder_parameters(encoder, &param);

    _newbitrate_ = (int)_newbitrate_;

    param.rc.i_bitrate = _newbitrate_;
    param.rc.i_vbv_max_bitrate = _newbitrate_;
    param.rc.i_vbv_buffer_size = param.rc.i_bitrate / RBR_BUFFER_SIZE_COEFF;

    x264_encoder_reconfig(encoder, &param);
}

void mythX264Encoder::Cleanup()
{
    x264_encoder_close(encoder);
    x264_picture_clean(&pic_in);
}

void mythX264Encoder::ProcessFrame(unsigned char **src, int *srclinesize, responseHandler *response)
{
    for (int i = 0; i < 3; i++)
    {
        pic_in.img.plane[i] = src[i];
        pic_in.img.i_stride[i] = srclinesize[i];
    }
    int len = x264_encoder_encode(encoder, &nal, &i_nal, &pic_in, &pic_out);
    if (len <= 0)
    {
        return;
    }
    else
    {
        for (int j = 0; j < i_nal; ++j)
        {
            response(this->hwnd, (char *)nal[j].p_payload, nal[j].i_payload);
        }
    }
}
