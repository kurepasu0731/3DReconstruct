#include "Graycode.h"
#include "Header.h"
#include "Calibration.h"


#define MASK_ADDRESS "./GrayCodeImage/mask.bmp"
#define IMAGE_DIRECTORY "./UseImage"
#define SAVE_DIRECTORY "./UseImage/resize"

void eular2rot(double yaw,double pitch, double roll, cv::Mat& dest);

//XMLファイル読み込み
std::vector<cv::Point3f> loadXMLfile(const std::string &fileName);

//--PLY保存系メソッド--//　//TODO:分ける
//PLY形式で保存
void savePLY(std::vector<cv::Point3f> points, const std::string &fileName);
void savePLY_with_normal(std::vector<cv::Point3f> points, std::vector<cv::Point3f> normals, const std::string &fileName);
void savePLY_with_normal_mesh(std::vector<cv::Point3f> points, std::vector<cv::Point3f> normals, std::vector<cv::Point3i> meshes, const std::string &fileName);

//--filter系メソッド--//　//TODO:分ける
//法線ベクトルを求める
std::vector<cv::Point3f> getNormalVectors(std::vector<cv::Point3f> points);
//ダウンサンプリング
std::vector<cv::Point3f> getDownSampledPoints(std::vector<cv::Point3f> points, float size);
//三角メッシュを生成
std::vector<cv::Point3i> getMeshVectors(std::vector<cv::Point3f> points, std::vector<cv::Point3f> normals);

int main()
{

	printf("0：キャリブレーションの読み込み\n");
	printf("1：背景取得\n");
	printf("2：対象の3次元復元\n");
	printf("3：メッシュの生成及びPLY形式での保存\n");
	printf("4: 取得済みデータ読み込みデータで背景差分\n");
	printf("5: 取得済みデータ読み込み,カメラから見た対象物体の画素にマスク(-2)をかける\n");
	printf("w：待機時に白画像を投影するかしないか\n");
	printf("\n");

	WebCamera webcamera(CAMERA_WIDTH, CAMERA_HEIGHT, "WebCamera");
	GRAYCODE gc(webcamera);

	// カメラ画像確認用
	char windowNameCamera[] = "camera";
	cv::namedWindow(windowNameCamera, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(windowNameCamera, 500, 300);

	static bool prjWhite = true;


	// キャリブレーション用
	Calibration calib(10, 7, 48.0);
	std::vector<std::vector<cv::Point3f>>	worldPoints;
	std::vector<std::vector<cv::Point2f>>	cameraPoints;
	std::vector<std::vector<cv::Point2f>>	projectorPoints;
	int calib_count = 0;

	//背景の閾値(mm)
	double thresh = 300.0;

	//背景と対象物の3次元点
	std::vector<cv::Point3f> reconstructPoint_back;
	std::vector<cv::Point3f> reconstructPoint_obj;

	// キー入力受付用の無限ループ
	while(true){
		printf("====================\n");
		printf("数字を入力してください....\n");
		int command;

		// 白い画像を全画面で投影（撮影環境を確認しやすくするため）
		cv::Mat cam, cam2;
		while(true){
			// trueで白を投影、falseで通常のディスプレイを表示
			if(prjWhite){
				cv::Mat white = cv::Mat(PROJECTOR_WIDTH, PROJECTOR_HEIGHT, CV_8UC3, cv::Scalar(255, 255, 255));
				cv::namedWindow("white_black", 0);
				Projection::MySetFullScrean(DISPLAY_NUMBER, "white_black");
				cv::imshow("white_black", white);
			}

			// 何かのキーが入力されたらループを抜ける
			command = cv::waitKey(33);
			if ( command > 0 ) break;

			cam = webcamera.getFrame();
			cam.copyTo(cam2);

			//見やすいように適当にリサイズ
			cv::resize(cam, cam, cv::Size(), 0.45, 0.45);
			cv::imshow(windowNameCamera, cam);
		}

		// カメラを止める
		cv::destroyWindow("white_black");

		// 条件分岐
		switch (command){

		case '0':

			std::cout << "キャリブレーション結果の読み込み中…" << std::endl;
			calib.loadCalibParam("calibration.xml");
				
			break;

		case '1':
			if(calib.calib_flag)
			{
				std::cout << "背景の3次元復元中…" << std::endl;

				// グレイコード投影
				gc.code_projection();
				gc.make_thresh();
				gc.makeCorrespondence();

				//***対応点の取得(カメラ画素→3次元点)******************************************
				std::vector<cv::Point2f> imagePoint_back;
				std::vector<cv::Point2f> projPoint_back;
				std::vector<int> isValid_back; //有効な対応点かどうかのフラグ
				//std::vector<cv::Point3f> reconstructPoint_back;
				gc.getCorrespondAllPoints_ProCam(projPoint_back, imagePoint_back, isValid_back);

				// 対応点の歪み除去
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

				// 3次元復元
				calib.reconstruction(reconstructPoint_back, undistort_projPoint_back, undistort_imagePoint_back, isValid_back);

				//==保存==//
				cv::FileStorage fs_back("./reconstructPoints_background.xml", cv::FileStorage::WRITE);
				write(fs_back, "points", reconstructPoint_back);
				std::cout << "background points saved." << std::endl;

				//**********************************************************************************

				// 描画
				cv::Mat R = cv::Mat::eye(3,3,CV_64F);
				cv::Mat t = cv::Mat::zeros(3,1,CV_64F);
				int key=0;
				cv::Point3d viewpoint(0.0,0.0,400.0);		// 視点位置
				cv::Point3d lookatpoint(0.0,0.0,0.0);	// 視線方向
				const double step = 50;

				// キーボード操作
				while(true)
				{
					//// 回転の更新
					double x=(lookatpoint.x-viewpoint.x);
					double y=(lookatpoint.y-viewpoint.y);
					double z=(lookatpoint.z-viewpoint.z);
					double pitch =asin(x/sqrt(x*x+z*z))/CV_PI*180.0;
					double yaw   =asin(-y/sqrt(y*y+z*z))/CV_PI*180.0;
					eular2rot(yaw, pitch, 0, R);
					// 移動の更新
					t.at<double>(0,0)=viewpoint.x;
					t.at<double>(1,0)=viewpoint.y;
					t.at<double>(2,0)=viewpoint.z;

					//カメラ画素→3次元点
					calib.pointCloudRender(reconstructPoint_back, cam2, std::string("viewer"), R, t);
					//calib.pointCloudRender(reconstructPoint_back, imagePoint_back, cam2, std::string("viewer"), R, t);

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
				std::cout << "キャリブレーションデータがありません" << std::endl;
			}

			break;

		case '2':
			if(calib.calib_flag)
			{
				std::cout << "対象の3次元復元中…" << std::endl;

				// グレイコード投影
				gc.code_projection();
				gc.make_thresh();
				gc.makeCorrespondence();

				//***対応点の取得(カメラ画素→3次元点)******************************************
				std::vector<cv::Point2f> imagePoint_obj;
				std::vector<cv::Point2f> projPoint_obj;
				std::vector<int> isValid_obj; //有効な対応点かどうかのフラグ
				//std::vector<cv::Point3f> reconstructPoint_obj;
				gc.getCorrespondAllPoints_ProCam(projPoint_obj, imagePoint_obj, isValid_obj);

				// 対応点の歪み除去
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

				// 3次元復元
				calib.reconstruction(reconstructPoint_obj, undistort_projPoint_obj, undistort_imagePoint_obj, isValid_obj);

				//==保存==//
				cv::FileStorage fs_obj("./reconstructPoints_obj.xml", cv::FileStorage::WRITE);
				write(fs_obj, "points", reconstructPoint_obj);
				std::cout << "object points saved." << std::endl;

				//**********************************************************************************

				// 描画
				cv::Mat R = cv::Mat::eye(3,3,CV_64F);
				cv::Mat t = cv::Mat::zeros(3,1,CV_64F);
				int key=0;
				cv::Point3d viewpoint(0.0,0.0,400.0);		// 視点位置
				cv::Point3d lookatpoint(0.0,0.0,0.0);	// 視線方向
				const double step = 50;

				// キーボード操作
				while(true)
				{
					//// 回転の更新
					double x=(lookatpoint.x-viewpoint.x);
					double y=(lookatpoint.y-viewpoint.y);
					double z=(lookatpoint.z-viewpoint.z);
					double pitch =asin(x/sqrt(x*x+z*z))/CV_PI*180.0;
					double yaw   =asin(-y/sqrt(y*y+z*z))/CV_PI*180.0;
					eular2rot(yaw, pitch, 0, R);
					// 移動の更新
					t.at<double>(0,0)=viewpoint.x;
					t.at<double>(1,0)=viewpoint.y;
					t.at<double>(2,0)=viewpoint.z;

					//カメラ画素→3次元点
					calib.pointCloudRender(reconstructPoint_obj, cam2, std::string("viewer"), R, t);
					//calib.pointCloudRender(reconstructPoint_obj, imagePoint_obj, cam2, std::string("viewer"), R, t);

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
				std::cout << "キャリブレーションデータがありません" << std::endl;
			}

			break;

		//PLY形式で保存
		case '3':
			{
				//有効な点のみ取りだす(= -1は除く)
				std::vector<cv::Point3f> validPoints;
				for(int n = 0; n < reconstructPoint_obj.size(); n++)
				{
					if(reconstructPoint_obj[n].x != -1) validPoints.emplace_back(cv::Point3f(reconstructPoint_obj[n].x/1000, reconstructPoint_obj[n].y/1000, reconstructPoint_obj[n].z/1000)); //単位をmに
				}

				//ダウンサンプリング
				std::vector<cv::Point3f> sampledPoints = getDownSampledPoints(validPoints, 0.01f);

				//法線を求める
				std::vector<cv::Point3f> normalVecs = getNormalVectors(sampledPoints);

				//メッシュを求める
				std::vector<cv::Point3i> meshes = getMeshVectors(sampledPoints, normalVecs);

				//PLY形式で保存
				savePLY_with_normal_mesh(sampledPoints, normalVecs, meshes, "reconstructPoint_obj_mesh.ply");
			}
			break;

		case '4':
			{
				reconstructPoint_back = loadXMLfile("reconstructPoints_background.xml");
				reconstructPoint_obj = loadXMLfile("reconstructPoints_obj.xml");

				//TODO:背景除去
				for(int i = 0; i < reconstructPoint_obj.size(); i++)
				{
					//閾値よりも深度の変化が小さかったら、(-1,-1,-1)で埋める
					if(reconstructPoint_obj[i].z == -1 || reconstructPoint_back[i].z == -1 || abs(reconstructPoint_obj[i].z - reconstructPoint_back[i].z) < thresh)
					{
						reconstructPoint_obj[i].x = -1;
						reconstructPoint_obj[i].y = -1;
						reconstructPoint_obj[i].z = -1;
					}
				}

				//==保存==//
				cv::FileStorage fs_obj("./reconstructPoints_obj_backremove.xml", cv::FileStorage::WRITE);
				write(fs_obj, "points", reconstructPoint_obj);
				std::cout << "back removed object points saved." << std::endl;

				// 描画
				cv::Mat R = cv::Mat::eye(3,3,CV_64F);
				cv::Mat t = cv::Mat::zeros(3,1,CV_64F);
				int key=0;
				cv::Point3d viewpoint(0.0,0.0,400.0);		// 視点位置
				cv::Point3d lookatpoint(0.0,0.0,0.0);	// 視線方向
				const double step = 50;

				// キーボード操作
				while(true)
				{
					//// 回転の更新
					double x=(lookatpoint.x-viewpoint.x);
					double y=(lookatpoint.y-viewpoint.y);
					double z=(lookatpoint.z-viewpoint.z);
					double pitch =asin(x/sqrt(x*x+z*z))/CV_PI*180.0;
					double yaw   =asin(-y/sqrt(y*y+z*z))/CV_PI*180.0;
					eular2rot(yaw, pitch, 0, R);
					// 移動の更新
					t.at<double>(0,0)=viewpoint.x;
					t.at<double>(1,0)=viewpoint.y;
					t.at<double>(2,0)=viewpoint.z;

					//カメラ画素→3次元点
					calib.pointCloudRender(reconstructPoint_obj, cam2, std::string("viewer"), R, t);

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


			}
			break;

		case '5':
			{
				reconstructPoint_back = loadXMLfile("reconstructPoints_background.xml");
				reconstructPoint_obj = loadXMLfile("reconstructPoints_obj.xml");

				for(int i = 0; i < reconstructPoint_obj.size(); i++)
				{
					//背景点群で(-1, -1, -1)だったところは、投影領域外
					if(reconstructPoint_back[i].z == -1 && reconstructPoint_obj[i].z == -1) continue;

					//投影領域内で深度が閾値以上になった部分は、投影対象物エリア、または影部分
					if(abs(reconstructPoint_obj[i].z - reconstructPoint_back[i].z) > thresh)
					{
						//影部分はobj点群では(-1, -1, -1)であるので、わかり、その部分は壁として残す
						//また、obj点群で値が取れていたとしても、差が閾値以下なら壁として残す？
						if(reconstructPoint_obj[i].z == -1)
						{
							reconstructPoint_obj[i].x = reconstructPoint_back[i].x;
							reconstructPoint_obj[i].y = reconstructPoint_back[i].y;
							reconstructPoint_obj[i].z = reconstructPoint_back[i].z;
						}
						else 
						{
							reconstructPoint_obj[i].x = -2;
							reconstructPoint_obj[i].y = -2;
							reconstructPoint_obj[i].z = -2;
						}
					}

				}

				//==保存==//
				cv::FileStorage fs_obj("./reconstructPoints_camera_mask.xml", cv::FileStorage::WRITE);
				write(fs_obj, "points", reconstructPoint_obj);
				std::cout << "back removed object points saved." << std::endl;

				// 描画
				cv::Mat R = cv::Mat::eye(3,3,CV_64F);
				cv::Mat t = cv::Mat::zeros(3,1,CV_64F);
				int key=0;
				cv::Point3d viewpoint(0.0,0.0,400.0);		// 視点位置
				cv::Point3d lookatpoint(0.0,0.0,0.0);	// 視線方向
				const double step = 50;

				// キーボード操作
				while(true)
				{
					//// 回転の更新
					double x=(lookatpoint.x-viewpoint.x);
					double y=(lookatpoint.y-viewpoint.y);
					double z=(lookatpoint.z-viewpoint.z);
					double pitch =asin(x/sqrt(x*x+z*z))/CV_PI*180.0;
					double yaw   =asin(-y/sqrt(y*y+z*z))/CV_PI*180.0;
					eular2rot(yaw, pitch, 0, R);
					// 移動の更新
					t.at<double>(0,0)=viewpoint.x;
					t.at<double>(1,0)=viewpoint.y;
					t.at<double>(2,0)=viewpoint.z;

					//カメラ画素→3次元点
					calib.pointCloudRender(reconstructPoint_obj, cam2, std::string("viewer"), R, t);

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


			}
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

//---保存関係---//

//PLY形式で保存(法線なし)
void savePLY(std::vector<cv::Point3f> points, const std::string &fileName)
{
	//重心を求める
	double cx = 0, cy = 0, cz = 0;
	for (int n = 0; n < points.size(); n++){
		cx += points[n].x;
		cy += points[n].y;
		cz += points[n].z;
	}
	cx /= points.size();
	cy /= points.size();
	cz /= points.size();

	//ファイルオープン
	FILE *fp;
	fp = fopen(fileName.data(), "w");

	//ファイルに書き込む
	//ヘッダの設定
	fprintf(fp,"ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n", points.size());

	//3次元点群
	//m単位で保存（xmlはmm）
	//重心を原点にする
	for (int n = 0; n < points.size(); n++){
	   fprintf(fp, "%f %f %f \n", (points[n].x - cx), (points[n].y - cy), (points[n].z - cz));
	}
	//ファイルクローズ
	fclose(fp);
}

//PLY形式で保存(法線あり)
void savePLY_with_normal(std::vector<cv::Point3f> points, std::vector<cv::Point3f> normals, const std::string &fileName)
{
	//重心を求める
	double cx = 0, cy = 0, cz = 0;
	for (int n = 0; n < points.size(); n++){
		cx += points[n].x;
		cy += points[n].y;
		cz += points[n].z;
	}
	cx /= points.size();
	cy /= points.size();
	cz /= points.size();

	//ファイルオープン
	FILE *fp;
	fp = fopen(fileName.data(), "w");

	//ファイルに書き込む
	//ヘッダの設定
	fprintf(fp,"ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nend_header\n", points.size());

	//3次元点群
	//m単位で保存（xmlはmm）
	//重心を原点にする
	for (int n = 0; n < points.size(); n++){
	   fprintf(fp, "%f %f %f %f %f %f \n", (points[n].x - cx), (points[n].y - cy), (points[n].z - cz), normals[n].x, normals[n].y, normals[n].z);
	}
	//ファイルクローズ
	fclose(fp);
}

//PLY形式で保存(法線あり,meshあり)
void savePLY_with_normal_mesh(std::vector<cv::Point3f> points, std::vector<cv::Point3f> normals, std::vector<cv::Point3i> meshes, const std::string &fileName)
{
	//重心を求める
	double cx = 0, cy = 0, cz = 0;
	for (int n = 0; n < points.size(); n++){
		cx += points[n].x;
		cy += points[n].y;
		cz += points[n].z;
	}
	cx /= points.size();
	cy /= points.size();
	cz /= points.size();

	//ファイルオープン
	FILE *fp;
	fp = fopen(fileName.data(), "w");

	//ファイルに書き込む
	//ヘッダの設定
	fprintf(fp,"ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nelement face %d\nproperty list ushort int vertex_indices\nend_header\n", points.size(), meshes.size());

	//3次元点群
	//m単位で保存（xmlはmm）
	//重心を原点にする
	for (int n = 0; n < points.size(); n++){
	   fprintf(fp, "%f %f %f %f %f %f \n", (points[n].x - cx), (points[n].y - cy), (points[n].z - cz), normals[n].x, normals[n].y, normals[n].z);
	}
	//面情報記述
	for(int n = 0; n < meshes.size(); n++)
	{
	   fprintf(fp, "3 %d %d %d\n", meshes[n].x, meshes[n].y, meshes[n].z);
	}
	//ファイルクローズ
	fclose(fp);
}

//---Filter関係---//

//法線ベクトルを求める
std::vector<cv::Point3f> getNormalVectors(std::vector<cv::Point3f> points)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	for (int n = 0; n < points.size(); n++){
		cloud->push_back(pcl::PointXYZ(points[n].x, points[n].y, points[n].z));//単位はm
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
		  //”1.#QNAN0”を0にする
		  if(!cvIsNaN(cloud_normals->at(n).normal_x))
			  dst_normals.emplace_back(cv::Point3f(cloud_normals->at(n).normal_x, cloud_normals->at(n).normal_y, cloud_normals->at(n).normal_z));
		  else
			  dst_normals.emplace_back(cv::Point3f(0, 0, 0));
	  }
	  return dst_normals;
}

//ダウンサンプリング
std::vector<cv::Point3f> getDownSampledPoints(std::vector<cv::Point3f> points, float size)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

	for (int n = 0; n < points.size(); n++){
		cloud->push_back(pcl::PointXYZ(points[n].x, points[n].y, points[n].z));//単位はm
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


//三角メッシュを生成、メッシュ情報を返す
std::vector<cv::Point3i> getMeshVectors(std::vector<cv::Point3f> points, std::vector<cv::Point3f> normals)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
	for (int n = 0; n < points.size(); n++){
		cloud->push_back(pcl::PointXYZ(points[n].x, points[n].y, points[n].z));//単位はm
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

	//メッシュ情報を返す
	std::vector<cv::Point3i> dst_meshes;
	for(int n = 0; n < triangles.polygons.size(); n++)
	{
		dst_meshes.emplace_back(cv::Point3i(triangles.polygons[n].vertices[0], triangles.polygons[n].vertices[1], triangles.polygons[n].vertices[2])); 
	}

	return dst_meshes;
}

// XMLファイル読み込み
std::vector<cv::Point3f> loadXMLfile(const std::string &fileName)
{
	//読み込む点群
	std::vector<cv::Point3f> reconstructPoints;
	// xmlファイルの読み込み
	cv::FileStorage cvfs(fileName, cv::FileStorage::READ);
	cvfs["points"] >> reconstructPoints;

	return reconstructPoints;
}
// オイラー角を行列に変換
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