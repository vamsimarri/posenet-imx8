#include <string>
#include <array>
#include <vector>

typedef unsigned char uint8_t;
// All Coral pose estimation models assume there are 17 keypoints per pose.
#define POSE_NUM_KEYPOINTS (17)
#define POSE_NUM_POSE_MAX (10)

// Defines dimension of an image, in height, width, depth order.
struct ImageDims {
  int height;
  int width;
  int depth;
};

struct Keypoint {
  float y;
  float x;
  float score;
};

struct Pose {
  float score;
  std::array<Keypoint, POSE_NUM_KEYPOINTS> keypoints;
};

class posenet_imx8{
public: 
	posenet_imx8(const std::string model_name);
	~posenet_imx8();
	std::vector<Pose> run_inference(const std::vector<uint8_t> &inputImage, const ImageDims& inputImageDims);
};
