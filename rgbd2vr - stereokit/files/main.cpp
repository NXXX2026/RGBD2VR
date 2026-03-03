#include <boost/asio.hpp>
#include <opencv2/opencv.hpp>

#include "stereokit.h"
#include "stereokit_ui.h"
#include "floor.hlsl.h"

using namespace boost::asio;
using boost::asio::ip::tcp;

cv::Mat ImgOut(720, 1280 * 2, CV_8UC3);

sk::model_t img_model;
sk::material_t img_material;
sk::tex_t img_tex;
sk::shader_t add_shader;

std::atomic<bool> should_update = false;
std::atomic<bool> should_stop_stream = false;

std::mutex img_mtx;

const char* focusStateToString(sk::app_focus_ focusState) {
	switch (focusState) {
	case sk::app_focus_active:      return "app_focus_active";
	case sk::app_focus_background:  return "app_focus_background";
	case sk::app_focus_hidden:      return "app_focus_hidden";
	default:                    return "unknown";
	}
}

/*
void receiver() {
	io_service io_service;
	tcp::resolver resolver(io_service);
	auto endpoint = resolver.resolve({ "localhost", "8080" });
	tcp::socket socket(io_service);
	connect(socket, endpoint);

	while (!should_stop_stream)
	{
		std::size_t total;
		boost::system::error_code ec;
		boost::asio::read(socket, boost::asio::buffer(&total, sizeof(total)), ec);

		std::vector<uchar> buffer(total);
		boost::asio::read(socket, boost::asio::buffer(buffer.data(), buffer.size()), ec);

		cv::Mat data_mat(buffer, true);
		cv::Mat temp = cv::imdecode(data_mat, 1);

		if (temp.empty()) {
			std::cout << "cv::imdecode failed!" << std::endl;
			continue;
		}

		img_mtx.lock();
		ImgOut = temp;
		img_mtx.unlock();

		if (should_stop_stream) {
			break;
		}
	}
}
*/

void receiver() {
	boost::asio::io_service io_service;
	boost::asio::ip::tcp::resolver resolver(io_service);
	auto endpoint = resolver.resolve({ "10.28.0.64", "8080" });
	boost::asio::ip::tcp::socket socket(io_service);

	// Animation characters
	std::string spinner = "/-\\|";
	int i = 0;

	// Loop until connection is established
	while (true) {
		try {
			boost::asio::connect(socket, endpoint);
			std::cout << "\nConnected.\n";
			break;  // Break out of the loop once connected
		}
		catch (boost::system::system_error& e) {
			// Connection failed, wait a bit before retrying
			std::this_thread::sleep_for(std::chrono::seconds(1));
			std::cout << "\rWaiting for connection " << spinner[i++ % spinner.size()] << std::flush;
		}
	}

	while (!should_stop_stream)
	{
		std::size_t total;
		boost::system::error_code ec;
		boost::asio::read(socket, boost::asio::buffer(&total, sizeof(total)), ec);

		std::vector<uchar> buffer(total);
		boost::asio::read(socket, boost::asio::buffer(buffer.data(), buffer.size()), ec);

		cv::Mat data_mat(buffer, true);
		cv::Mat temp = cv::imdecode(data_mat, 1);

		if (temp.empty()) {
			std::cout << "cv::imdecode failed!" << std::endl;
			break;
		}

		img_mtx.lock();
		ImgOut = temp;
		img_mtx.unlock();

		if (should_stop_stream) {
			break;
		}
	}
}

void check_focus() {
	sk::app_focus_ initialFocusState = sk::sk_app_focus();
	sk::app_focus_ prevFocusState = initialFocusState;
	bool first_transition_occurred = false;

	while (true && !should_stop_stream) {
		sk::app_focus_ currentFocusState = sk::sk_app_focus();

		if (should_stop_stream) {
			break;
		}

		if (currentFocusState != prevFocusState) {
			printf("\nFocus state changed: %s -> %s\n\n", focusStateToString(prevFocusState), focusStateToString(currentFocusState));
			prevFocusState = currentFocusState;

			if (!first_transition_occurred && currentFocusState == sk::app_focus_active) {
				first_transition_occurred = true;
			}
		}

		if (!first_transition_occurred && currentFocusState == sk::app_focus_active) {
			first_transition_occurred = true;
			should_update = true;
		}

		if (first_transition_occurred) {
			if (currentFocusState == sk::app_focus_background || currentFocusState == sk::app_focus_hidden) {
				should_update = false;
			}
			else if (currentFocusState == sk::app_focus_active) {
				should_update = true;
			}
		}
	}
}

cv::Mat addAlphaChannel(cv::Mat& input) {
	// Check if the input image has 3 channels (BGR)
	if (input.channels() != 3) {
		std::cerr << "The input image must have 3 channels (BGR).\n";
		return cv::Mat();
	}

	std::vector<cv::Mat> channels(3);
	cv::split(input, channels);  // split input into the B, G, R channels

	// Create an alpha channel
	cv::Mat alpha(input.rows, input.cols, CV_8UC1, cv::Scalar(255));  // this creates a white (all 255) alpha channel
	channels.push_back(alpha);  // add alpha channel to the channels vector

	cv::Mat output;
	cv::merge(channels, output);  // merge all four channels into the output image

	return output;
}

void update()
{
	if (!should_update) {
		return;
	}

	int img_rows;
	int img_cols;
	uchar* img_data;
	{
		std::lock_guard<std::mutex> lock(img_mtx); // Lock the mutex
		if (ImgOut.data == NULL) {
			return;
		}

		if (img_tex == NULL) {
			std::cerr << "img_tex is not valid!\n";
			return;
		}

		if (ImgOut.rows <= 0 || ImgOut.cols <= 0) {
			std::cerr << "Invalid image dimensions!\n";
			return;
		}

		cv::Mat newMat = addAlphaChannel(ImgOut);

		// Make copies of the needed data while still in the mutex-protected block.
		img_rows = newMat.rows;
		img_cols = newMat.cols;
		img_data = newMat.data;

		sk::tex_set_colors(img_tex, img_cols, img_rows, img_data);
		sk::material_set_texture(img_material, "diffuse", img_tex);
	}

	// Now we can use the copies we made. No need to access ImgOut again.
	float height = 0.5f;
	sk::vec3 offset = { 0, 0, -2.0 };
	sk::vec3 scale = { (float)img_cols / (float)img_rows * height, height, 1 };
	sk::vec3 user_position = sk::input_head()->position;
	sk::vec3 model_position = user_position + sk::vec3{ 0,0,-3 };

	sk::render_add_model(img_model, matrix_trs(model_position, sk::quat_identity, scale));
}

/*
void update()
{
	if (!should_update) {
		return;
	}

	std::unique_lock<std::mutex> lock(our_man.get()->mutex, std::try_to_lock); // Use std::unique_lock with std::mutex and std::try_to_lock
	if (lock.owns_lock()) {
		if (our_man.get()->front_mat != NULL) {
			std::cout << "The cols is: " << our_man.get()->front_mat->cols << std::endl;
			std::cout << "The rows is: " << our_man.get()->front_mat->rows << std::endl;
			std::cout << "The number of channels is: " << our_man.get()->front_mat->channels() << std::endl;
			sk::tex_set_colors(img_tex, our_man.get()->front_mat->cols, our_man.get()->front_mat->rows,
				our_man.get()->front_mat->data);
			sk::material_set_texture(img_material, "diffuse", img_tex);
		}
	}
	float height = 0.5f;
	sk::vec3 offset = { 0, 0, -2.0 };
	sk::vec3 scale = { our_man.get()->width_over_height * height, height, 1 };
	sk::vec3 user_position = sk::input_head()->position;
	sk::vec3 model_position = user_position + sk::vec3{ 0,0,-3 };

	sk::render_add_model(img_model, matrix_trs(model_position, sk::quat_identity, scale));
}
*/

int main()
{
	std::thread receive_thread(receiver);
	sk::sk_settings_t settings = {};
	settings.app_name = "VR demo!";
	settings.display_preference = sk::display_mode_mixedreality;
	settings.assets_folder = "Assets";
	if (!sk_init(settings))
		return 1;

	sk::mesh_t m = sk::mesh_gen_plane({ 2,4 }, { 0, 0, 1 }, { 0, 1, 0 });

	img_tex = sk::tex_create(sk::tex_type_image, sk::tex_format_rgba32);

	sk::tex_set_sample(img_tex, sk::tex_sample_point);

	add_shader = sk::shader_create_mem((void*)sks_floor_hlsl, sizeof(sks_floor_hlsl));
	img_material = sk::material_create(add_shader);

	sk::material_set_float(img_material, "tex_scale", 1);
	material_set_cull(img_material, sk::cull_none);

	img_model = model_create_mesh(m, img_material);

	std::thread focus_thread(check_focus);

	sk::sk_run([]() {
		update();
	});
	
	should_stop_stream = true;

	focus_thread.join();
	receive_thread.join();

	std::cout << "\nProgram execution completed successfully. Exiting..." << std::endl;
	return 0;
}
