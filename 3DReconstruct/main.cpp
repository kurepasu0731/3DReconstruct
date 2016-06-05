#include "Graycode.h"
#include "Header.h"
#include "Calibration.h"


#define MASK_ADDRESS "./GrayCodeImage/mask.bmp"
#define IMAGE_DIRECTORY "./UseImage"
#define SAVE_DIRECTORY "./UseImage/resize"

void eular2rot(double yaw,double pitch, double roll, cv::Mat& dest);

//�w�i��臒l(mm)
double thresh = 5.0;

//�w�i��3�����_
std::vector<cv::Point3f> reconstructPoint_back;
std::vector<cv::Point3f> reconstructPoint_obj;

//--PLY�ۑ��n���\�b�h--//
//PLY�`���ŕۑ�
void savePLY(std::vector<cv::Point3f> points, const std::string &fileName);
void savePLY_with_normal(std::vector<cv::Point3f> points, std::vector<cv::Point3f> normals, const std::string &fileName);
void savePLY_with_normal_mesh(std::vector<cv::Point3f> points, std::vector<cv::Point3f> normals, std::vector<cv::Point3i> meshes, const std::string &fileName);

//�@���x�N�g�������߂�
std::vector<cv::Point3f> getNormalVectors(std::vector<cv::Point3f> points);
//�_�E���T���v�����O
std::vector<cv::Point3f> getDownSampledPoints(std::vector<cv::Point3f> points, float size);
//�O�p���b�V���𐶐�
std::vector<cv::Point3i> getMeshVectors(std::vector<cv::Point3f> points, std::vector<cv::Point3f> normals);

int main()
{

	printf("0�F�L�����u���[�V�����̓ǂݍ���\n");
	printf("1�F�w�i�擾\n");
	printf("2�F�Ώۂ�3��������\n");
	printf("3�F���b�V���̐����y��PLY�`���ł̕ۑ�\n");


	printf("w�F�ҋ@���ɔ��摜�𓊉e���邩���Ȃ���\n");
	printf("\n");

	WebCamera webcamera(CAMERA_WIDTH, CAMERA_HEIGHT, "WebCamera");
	GRAYCODE gc(webcamera);

	// �J�����摜�m�F�p
	char windowNameCamera[] = "camera";
	cv::namedWindow(windowNameCamera, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(windowNameCamera, 500, 300);

	static bool prjWhite = true;


	// �L�����u���[�V�����p
	Calibration calib(10, 7, 48.0);
	std::vector<std::vector<cv::Point3f>>	worldPoints;
	std::vector<std::vector<cv::Point2f>>	cameraPoints;
	std::vector<std::vector<cv::Point2f>>	projectorPoints;
	int calib_count = 0;


	// �L�[���͎�t�p�̖������[�v
	while(true){
		printf("====================\n");
		printf("��������͂��Ă�������....\n");
		int command;

		// �����摜��S��ʂœ��e�i�B�e�����m�F���₷�����邽�߁j
		cv::Mat cam, cam2;
		while(true){
			// true�Ŕ��𓊉e�Afalse�Œʏ�̃f�B�X�v���C��\��
			if(prjWhite){
				cv::Mat white = cv::Mat(PROJECTOR_WIDTH, PROJECTOR_HEIGHT, CV_8UC3, cv::Scalar(255, 255, 255));
				cv::namedWindow("white_black", 0);
				Projection::MySetFullScrean(DISPLAY_NUMBER, "white_black");
				cv::imshow("white_black", white);
			}

			// �����̃L�[�����͂��ꂽ�烋�[�v�𔲂���
			command = cv::waitKey(33);
			if ( command > 0 ) break;

			cam = webcamera.getFrame();
			cam.copyTo(cam2);

			//���₷���悤�ɓK���Ƀ��T�C�Y
			cv::resize(cam, cam, cv::Size(), 0.45, 0.45);
			cv::imshow(windowNameCamera, cam);
		}

		// �J�������~�߂�
		cv::destroyWindow("white_black");

		// ��������
		switch (command){

		case '0':

			std::cout << "�L�����u���[�V�������ʂ̓ǂݍ��ݒ��c" << std::endl;
			calib.loadCalibParam("calibration.xml");
				
			break;

		case '1':
			if(calib.calib_flag)
			{
				std::cout << "�w�i��3�����������c" << std::endl;

				// �O���C�R�[�h���e
				gc.code_projection();
				gc.make_thresh();
				gc.makeCorrespondence();

				//***�Ή��_�̎擾(�J������f��3�����_)******************************************
				std::vector<cv::Point2f> imagePoint_back;
				std::vector<cv::Point2f> projPoint_back;
				std::vector<int> isValid_back; //�L���ȑΉ��_���ǂ����̃t���O
				//std::vector<cv::Point3f> reconstructPoint_back;
				gc.getCorrespondAllPoints_ProCam(projPoint_back, imagePoint_back, isValid_back);

				// �Ή��_�̘c�ݏ���
				std::vector<cv::Point2f> undistort_imagePoint_back;
				std::vector<cv::Point2f> undistort_projPoint_back;
				cv::undistortPoints(imagePoint_back, undistort_imagePoint_back, calib.cam_K, calib.cam_dist);
				cv::undistortPoints(projPoint_back, undistort_projPoint_back, calib.proj_K, calib.proj_dist);
				for(int i=0; i<imagePoint_back.size(); ++i)
				{
					if(isValid_back[i] == 1)
					{
						undistort_imagePoint_back[i].x = undistort_imagePoint_back[i].x * calib.cam_K.at<double>(0,0) + calib.cam_K.at<double>(0,2);
						undistort_imagePoint_back[i].y = undistort_imagePoint_back[i].y * calib.cam_K.at<double>(1,1) + calib.cam_K.at<double>(1,2);
						undistort_projPoint_back[i].x = undistort_projPoint_back[i].x * calib.proj_K.at<double>(0,0) + calib.proj_K.at<double>(0,2);
						undistort_projPoint_back[i].y = undistort_projPoint_back[i].y * calib.proj_K.at<double>(1,1) + calib.proj_K.at<double>(1,2);
					}
					else
					{
						undistort_imagePoint_back[i].x = -1;
						undistort_imagePoint_back[i].y = -1;
						undistort_projPoint_back[i].x = -1;
						undistort_projPoint_back[i].y = -1;
					}
				}

				// 3��������
				calib.reconstruction(reconstructPoint_back, undistort_projPoint_back, undistort_imagePoint_back, isValid_back);

				//==�ۑ�==//
				cv::FileStorage fs_back("./reconstructPoints_background.xml", cv::FileStorage::WRITE);
				write(fs_back, "points", reconstructPoint_back);
				std::cout << "background points saved." << std::endl;

				//**********************************************************************************

				// �`��
				cv::Mat R = cv::Mat::eye(3,3,CV_64F);
				cv::Mat t = cv::Mat::zeros(3,1,CV_64F);
				int key=0;
				cv::Point3d viewpoint(0.0,0.0,400.0);		// ���_�ʒu
				cv::Point3d lookatpoint(0.0,0.0,0.0);	// ��������
				const double step = 50;

				// �L�[�{�[�h����
				while(true)
				{
					//// ��]�̍X�V
					double x=(lookatpoint.x-viewpoint.x);
					double y=(lookatpoint.y-viewpoint.y);
					double z=(lookatpoint.z-viewpoint.z);
					double pitch =asin(x/sqrt(x*x+z*z))/CV_PI*180.0;
					double yaw   =asin(-y/sqrt(y*y+z*z))/CV_PI*180.0;
					eular2rot(yaw, pitch, 0, R);
					// �ړ��̍X�V
					t.at<double>(0,0)=viewpoint.x;
					t.at<double>(1,0)=viewpoint.y;
					t.at<double>(2,0)=viewpoint.z;

					//�J������f��3�����_
					calib.pointCloudRender(reconstructPoint_back, imagePoint_back, cam2, std::string("viewer"), R, t);

					key = cv::waitKey(0);
					if(key=='w')
					{
						viewpoint.y+=step;
					}
					if(key=='s')
					{
						viewpoint.y-=step;
					}
					if(key=='a')
					{
						viewpoint.x+=step;
					}
					if(key=='d')
					{
						viewpoint.x-=step;
					}
					if(key=='z')
					{
						viewpoint.z+=step;
					}
					if(key=='x')
					{
						viewpoint.z-=step;
					}
					if(key=='q')
					{
						break;
					}
				}

			} else {
				std::cout << "�L�����u���[�V�����f�[�^������܂���" << std::endl;
			}

			break;

		case '2':
			if(calib.calib_flag)
			{
				std::cout << "�Ώۂ�3�����������c" << std::endl;

				// �O���C�R�[�h���e
				gc.code_projection();
				gc.make_thresh();
				gc.makeCorrespondence();

				//***�Ή��_�̎擾(�J������f��3�����_)******************************************
				std::vector<cv::Point2f> imagePoint_obj;
				std::vector<cv::Point2f> projPoint_obj;
				std::vector<int> isValid_obj; //�L���ȑΉ��_���ǂ����̃t���O
				//std::vector<cv::Point3f> reconstructPoint_obj;
				gc.getCorrespondAllPoints_ProCam(projPoint_obj, imagePoint_obj, isValid_obj);

				// �Ή��_�̘c�ݏ���
				std::vector<cv::Point2f> undistort_imagePoint_obj;
				std::vector<cv::Point2f> undistort_projPoint_obj;
				cv::undistortPoints(imagePoint_obj, undistort_imagePoint_obj, calib.cam_K, calib.cam_dist);
				cv::undistortPoints(projPoint_obj, undistort_projPoint_obj, calib.proj_K, calib.proj_dist);
				for(int i=0; i<imagePoint_obj.size(); ++i)
				{
					if(isValid_obj[i] == 1)
					{
						undistort_imagePoint_obj[i].x = undistort_imagePoint_obj[i].x * calib.cam_K.at<double>(0,0) + calib.cam_K.at<double>(0,2);
						undistort_imagePoint_obj[i].y = undistort_imagePoint_obj[i].y * calib.cam_K.at<double>(1,1) + calib.cam_K.at<double>(1,2);
						undistort_projPoint_obj[i].x = undistort_projPoint_obj[i].x * calib.proj_K.at<double>(0,0) + calib.proj_K.at<double>(0,2);
						undistort_projPoint_obj[i].y = undistort_projPoint_obj[i].y * calib.proj_K.at<double>(1,1) + calib.proj_K.at<double>(1,2);
					}
					else
					{
						undistort_imagePoint_obj[i].x = -1;
						undistort_imagePoint_obj[i].y = -1;
						undistort_projPoint_obj[i].x = -1;
						undistort_projPoint_obj[i].y = -1;
					}
				}

				// 3��������
				calib.reconstruction(reconstructPoint_obj, undistort_projPoint_obj, undistort_imagePoint_obj, isValid_obj);

				//TODO:�w�i����
				for(int i = 0; i < reconstructPoint_obj.size(); i++)
				{
					//臒l�����[�x�̕ω���������������A(-1,-1,-1)�Ŗ��߂�
					if(abs(reconstructPoint_obj[i].z - reconstructPoint_back[i].z) < thresh)
					{
						reconstructPoint_obj[i].x = -1;
						reconstructPoint_obj[i].y = -1;
						reconstructPoint_obj[i].z = -1;
					}
				}

				//==�ۑ�==//
				cv::FileStorage fs_obj("./reconstructPoints_obj.xml", cv::FileStorage::WRITE);
				write(fs_obj, "points", reconstructPoint_obj);
				std::cout << "background points saved." << std::endl;

				//**********************************************************************************

				// �`��
				cv::Mat R = cv::Mat::eye(3,3,CV_64F);
				cv::Mat t = cv::Mat::zeros(3,1,CV_64F);
				int key=0;
				cv::Point3d viewpoint(0.0,0.0,400.0);		// ���_�ʒu
				cv::Point3d lookatpoint(0.0,0.0,0.0);	// ��������
				const double step = 50;

				// �L�[�{�[�h����
				while(true)
				{
					//// ��]�̍X�V
					double x=(lookatpoint.x-viewpoint.x);
					double y=(lookatpoint.y-viewpoint.y);
					double z=(lookatpoint.z-viewpoint.z);
					double pitch =asin(x/sqrt(x*x+z*z))/CV_PI*180.0;
					double yaw   =asin(-y/sqrt(y*y+z*z))/CV_PI*180.0;
					eular2rot(yaw, pitch, 0, R);
					// �ړ��̍X�V
					t.at<double>(0,0)=viewpoint.x;
					t.at<double>(1,0)=viewpoint.y;
					t.at<double>(2,0)=viewpoint.z;

					//�J������f��3�����_
					calib.pointCloudRender(reconstructPoint_obj, imagePoint_obj, cam2, std::string("viewer"), R, t);

					key = cv::waitKey(0);
					if(key=='w')
					{
						viewpoint.y+=step;
					}
					if(key=='s')
					{
						viewpoint.y-=step;
					}
					if(key=='a')
					{
						viewpoint.x+=step;
					}
					if(key=='d')
					{
						viewpoint.x-=step;
					}
					if(key=='z')
					{
						viewpoint.z+=step;
					}
					if(key=='x')
					{
						viewpoint.z-=step;
					}
					if(key=='q')
					{
						break;
					}
				}

			} else {
				std::cout << "�L�����u���[�V�����f�[�^������܂���" << std::endl;
			}

			break;

		//PLY�`���ŕۑ�
		case '3':
			//�L���ȓ_�̂ݎ�肾��(= -1�͏���)
			std::vector<cv::Point3f> validPoints;
			for(int n = 0; n < reconstructPoint_obj.size(); n++)
			{
				if(reconstructPoint_obj[n].x != -1) validPoints.emplace_back(cv::Point3f(reconstructPoint_obj[n].x/1000, reconstructPoint_obj[n].y/1000, reconstructPoint_obj[n].z/1000)); //�P�ʂ�m��
			}

			//�_�E���T���v�����O
			std::vector<cv::Point3f> sampledPoints = getDownSampledPoints(validPoints, 0.01f);

			//�@�������߂�
			std::vector<cv::Point3f> normalVecs = getNormalVectors(sampledPoints);

			//���b�V�������߂�
			std::vector<cv::Point3i> meshes = getMeshVectors(sampledPoints, normalVecs);

			//PLY�`���ŕۑ�
			savePLY_with_normal_mesh(sampledPoints, normalVecs, meshes, "reconstructPoint_obj_mesh.ply");
			break;

		case 'w':
			prjWhite = !prjWhite;
			break;

		default:
			exit(0);
			break;
		}
		printf("\n");
		cv::destroyAllWindows();
	}
}

//---�ۑ��֌W---//

//PLY�`���ŕۑ�(�@���Ȃ�)
void savePLY(std::vector<cv::Point3f> points, const std::string &fileName)
{
	//�d�S�����߂�
	double cx = 0, cy = 0, cz = 0;
	for (int n = 0; n < points.size(); n++){
		cx += points[n].x;
		cy += points[n].y;
		cz += points[n].z;
	}
	cx /= points.size();
	cy /= points.size();
	cz /= points.size();

	//�t�@�C���I�[�v��
	FILE *fp;
	fp = fopen(fileName.data(), "w");

	//�t�@�C���ɏ�������
	//�w�b�_�̐ݒ�
	fprintf(fp,"ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n", points.size());

	//3�����_�Q
	//m�P�ʂŕۑ��ixml��mm�j
	//�d�S�����_�ɂ���
	for (int n = 0; n < points.size(); n++){
	   fprintf(fp, "%f %f %f \n", (points[n].x - cx), (points[n].y - cy), (points[n].z - cz));
	}
	//�t�@�C���N���[�Y
	fclose(fp);
}

//PLY�`���ŕۑ�(�@������)
void savePLY_with_normal(std::vector<cv::Point3f> points, std::vector<cv::Point3f> normals, const std::string &fileName)
{
	//�d�S�����߂�
	double cx = 0, cy = 0, cz = 0;
	for (int n = 0; n < points.size(); n++){
		cx += points[n].x;
		cy += points[n].y;
		cz += points[n].z;
	}
	cx /= points.size();
	cy /= points.size();
	cz /= points.size();

	//�t�@�C���I�[�v��
	FILE *fp;
	fp = fopen(fileName.data(), "w");

	//�t�@�C���ɏ�������
	//�w�b�_�̐ݒ�
	fprintf(fp,"ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nend_header\n", points.size());

	//3�����_�Q
	//m�P�ʂŕۑ��ixml��mm�j
	//�d�S�����_�ɂ���
	for (int n = 0; n < points.size(); n++){
	   fprintf(fp, "%f %f %f %f %f %f \n", (points[n].x - cx), (points[n].y - cy), (points[n].z - cz), normals[n].x, normals[n].y, normals[n].z);
	}
	//�t�@�C���N���[�Y
	fclose(fp);
}

//PLY�`���ŕۑ�(�@������,mesh����)
void savePLY_with_normal_mesh(std::vector<cv::Point3f> points, std::vector<cv::Point3f> normals, std::vector<cv::Point3i> meshes, const std::string &fileName)
{
	//�d�S�����߂�
	double cx = 0, cy = 0, cz = 0;
	for (int n = 0; n < points.size(); n++){
		cx += points[n].x;
		cy += points[n].y;
		cz += points[n].z;
	}
	cx /= points.size();
	cy /= points.size();
	cz /= points.size();

	//�t�@�C���I�[�v��
	FILE *fp;
	fp = fopen(fileName.data(), "w");

	//�t�@�C���ɏ�������
	//�w�b�_�̐ݒ�
	fprintf(fp,"ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nelement face %d\nproperty list uchar int vertex_indices\nend_header\n", points.size(), meshes.size());

	//3�����_�Q
	//m�P�ʂŕۑ��ixml��mm�j
	//�d�S�����_�ɂ���
	for (int n = 0; n < points.size(); n++){
	   fprintf(fp, "%f %f %f %f %f %f \n", (points[n].x - cx), (points[n].y - cy), (points[n].z - cz), normals[n].x, normals[n].y, normals[n].z);
	}
	//�ʏ��L�q
	for(int n = 0; n < meshes.size(); n++)
	{
	   fprintf(fp, "3%d %d %d\n", meshes[n].x, meshes[n].y, meshes[n].z);
	}
	//�t�@�C���N���[�Y
	fclose(fp);
}

//---Filter�֌W---//

//�@���x�N�g�������߂�
std::vector<cv::Point3f> getNormalVectors(std::vector<cv::Point3f> points)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	for (int n = 0; n < points.size(); n++){
		cloud->push_back(pcl::PointXYZ(points[n].x, points[n].y, points[n].z));//�P�ʂ�m
	}

	  // Create the normal estimation class, and pass the input dataset to it
	  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	  ne.setInputCloud (cloud);

	  // Create an empty kdtree representation, and pass it to the normal estimation object.
	  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
	  ne.setSearchMethod (tree);

	  // Output datasets
	  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

	  // Use all neighbors in a sphere of radius 3cm
	  ne.setRadiusSearch (0.03);

	  // Compute the features
	  ne.compute (*cloud_normals);

	  std::vector<cv::Point3f> dst_normals;
	  for(int n = 0; n < cloud_normals->size(); n++)
	  {
		  //�h1.#QNAN0�h��0�ɂ���
		  if(!cvIsNaN(cloud_normals->at(n).normal_x))
			  dst_normals.emplace_back(cv::Point3f(cloud_normals->at(n).normal_x, cloud_normals->at(n).normal_y, cloud_normals->at(n).normal_z));
		  else
			  dst_normals.emplace_back(cv::Point3f(0, 0, 0));
	  }
	  return dst_normals;
}

//�_�E���T���v�����O
std::vector<cv::Point3f> getDownSampledPoints(std::vector<cv::Point3f> points, float size)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

	for (int n = 0; n < points.size(); n++){
		cloud->push_back(pcl::PointXYZ(points[n].x, points[n].y, points[n].z));//�P�ʂ�m
	}

    // Create the filtering object
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud (cloud);
	sor.setLeafSize (size, size, size);
	sor.filter (*cloud_filtered);

	std::vector<cv::Point3f> dst_points;
	for(int n = 0; n < cloud_filtered->size(); n++)
		dst_points.emplace_back(cv::Point3f(cloud_filtered->at(n).x, cloud_filtered->at(n).y, cloud_filtered->at(n).z));

	return dst_points;
}


//�O�p���b�V���𐶐��A���b�V������Ԃ�
std::vector<cv::Point3i> getMeshVectors(std::vector<cv::Point3f> points, std::vector<cv::Point3f> normals)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
	for (int n = 0; n < points.size(); n++){
		cloud->push_back(pcl::PointXYZ(points[n].x, points[n].y, points[n].z));//�P�ʂ�m
		cloud_normals->push_back(pcl::Normal(normals[n].x, normals[n].y, normals[n].z));
	}
	// Concatenate the XYZ and normal fields*
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields (*cloud, *cloud_normals, *cloud_with_normals);

	// Create search tree*
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud (cloud_with_normals);

	// Initialize objects
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	pcl::PolygonMesh triangles;

	// Set the maximum distance between connected points (maximum edge length)
	gp3.setSearchRadius (0.025);

	// Set typical values for the parameters
	gp3.setMu (2.5);
	gp3.setMaximumNearestNeighbors (100);
	gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
	gp3.setMinimumAngle(M_PI/18); // 10 degrees
	gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
	gp3.setNormalConsistency(false);

	// Get result
	gp3.setInputCloud (cloud_with_normals);
	gp3.setSearchMethod (tree2);
	gp3.reconstruct (triangles);

	//���b�V������Ԃ�
	std::vector<cv::Point3i> dst_meshes;
	for(int n = 0; n < triangles.polygons.size(); n++)
	{
		dst_meshes.emplace_back(cv::Point3i(triangles.polygons[n].vertices[0], triangles.polygons[n].vertices[1], triangles.polygons[n].vertices[2])); 
	}

	return dst_meshes;
}

// �I�C���[�p���s��ɕϊ�
void eular2rot(double yaw,double pitch, double roll, cv::Mat& dest)
{
    double theta = yaw/180.0*CV_PI;
    double pusai = pitch/180.0*CV_PI;
    double phi = roll/180.0*CV_PI;
 
    double datax[3][3] = {{1.0,0.0,0.0}, 
    {0.0,cos(theta),-sin(theta)}, 
    {0.0,sin(theta),cos(theta)}};
    double datay[3][3] = {{cos(pusai),0.0,sin(pusai)}, 
    {0.0,1.0,0.0}, 
    {-sin(pusai),0.0,cos(pusai)}};
    double dataz[3][3] = {{cos(phi),-sin(phi),0.0}, 
    {sin(phi),cos(phi),0.0}, 
    {0.0,0.0,1.0}};

    cv::Mat Rx(3,3,CV_64F,datax);
    cv::Mat Ry(3,3,CV_64F,datay);
    cv::Mat Rz(3,3,CV_64F,dataz);
    cv::Mat rr=Rz*Rx*Ry;

    rr.copyTo(dest);
}