#include "mythVirtualDecoder.hh"


mythVirtualDecoder::mythVirtualDecoder(void)
	:mythAvlist()
{
	flag = 0;
}
void mythVirtualDecoder::start()
{
}
void mythVirtualDecoder::stop()
{
}
mythVirtualDecoder* mythVirtualDecoder::CreateNew(void){
	return new mythVirtualDecoder();
}
mythVirtualDecoder::~mythVirtualDecoder(void)
{
}
