#include <cmath>
#include "posenet_imx8.h"
#include "posenet.h"

posenet_t* inference;

posenet_imx8::posenet_imx8(const std::string model_name){
	inference = new posenet_t ();
	//Initialize the posenet inference with model name
	inference->init(model_name, 2, 4);
}

posenet_imx8::~posenet_imx8(){
	delete inference;
	inference = NULL;
}

std::vector<Pose> posenet_imx8::run_inference(const std::vector<uint8_t> &inputImage, const ImageDims& originalImgDims)
{	
	inference->setup_input_tensor(originalImgDims.height, originalImgDims.width, originalImgDims.depth, (uint8_t*)inputImage.data());
 	inference->inference();

	float *keypoint_coord = inference->getKeypointCoords();
	float *keypoint_score = inference->getKeypointScores();
	float *pose_score = inference->getPoseScores();
	float* npose_f = inference->getNumPoses();
	int num_poses = (int)(*npose_f);

	std::vector<Pose> poses(num_poses);
	for (int i = 0; i < num_poses; ++i) {
		poses[i].score = pose_score[i];
		for (int k = 0; k < POSE_NUM_KEYPOINTS; ++k) { 
		poses[i].keypoints[k].score = *keypoint_score++;
		poses[i].keypoints[k].y = *keypoint_coord++;
		poses[i].keypoints[k].x = *keypoint_coord++;
		}
	}
	
	return poses;
}



