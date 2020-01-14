#include "MythConfig.hh"
#include "VirtualP2PServer.hh"
int main(int argc, char *argv[])
{
	//mythCameraDecoder* cameradecoder = mythCameraDecoder::CreateNew();
	mythCamera *camera = mythCamera::CreateNew();
	VirtualP2PServer* server = VirtualP2PServer::CreateNew();
	server->Start();
	return 0;
}
