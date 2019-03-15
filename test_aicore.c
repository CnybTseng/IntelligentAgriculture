#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#ifdef __linux__
#	include <termios.h>
#	include <sys/types.h>
#	include <unistd.h>
#	include <fcntl.h>
#endif
#ifdef _WIN32
#	include <conio.h>
#endif
#include "aicore.h"
#include "bitmap.h"

static int quit = 0;	// 退出应用程序的标志
static bitmap *bmp = NULL;

void print_help();
int create_aicore_test_thread(pthread_t *tid);
void *aicore_test_thread(void *param);
void draw_bounding_box(bitmap *bmp, int x, int y, int w, int h);
void wait_for_thread_dead(pthread_t tid);
#ifdef __linux__
int kbhit(void);
#endif

int main(int argc, char *argv[])
{
	if (argc < 2){
		print_help();
		return 0;
	}
		
	bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp fail[%s:%d].\n", __FILE__, __LINE__);
		return 0;
	}
	
	pthread_t tid;
	int keyboard = 0;
	const int width = get_bmp_width(bmp);
	const int height = get_bmp_height(bmp);
	const int pitch = get_bmp_pitch(bmp);
	const char *data = (char *)get_bmp_data(bmp);
	printf("bitmap width %d, height %d, pitch %d\n", width, height, pitch);
	
	int ret = ai_core_init(width, height);
	if (ret != AIC_OK) {
		fprintf(stderr, "ai_core_init fail, error code %d.\n", ret);
		goto cleanup;
	}
	
	if (create_aicore_test_thread(&tid)) goto cleanup;

	// const struct timespec req = {0, 1000000};
	while (!quit) {
#ifdef __linux__
		if (kbhit()) {
			keyboard = getchar();
#else
		if (_kbhit()) {
			keyboard = _getch();
#endif
			if (keyboard == 'q') {
				quit = 1;
				break;
			}
		}

		int ret = ai_core_send_image(data, width * height);
		if (ret != AIC_OK) {
			printf("ai_core_send_image fail! error code %d.\n", ret);
		}
	
		// nanosleep(&req, NULL); // 如果ai_core_send_image很快返回,此处需要延时,让算法有足够时间消耗队列中的数据
	}
	
	cleanup:
	wait_for_thread_dead(tid);	// 等待接收物体坐标的线程结束
	if (bmp) delete_bmp(bmp);
	ai_core_free();
	
	return 0;
}

void print_help()
{
	printf("Usage:\n");
	printf("      test_aicore [bitmap filename]\n");
}

int create_aicore_test_thread(pthread_t *tid)
{
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	
	int ret = pthread_create(tid, &attr, aicore_test_thread, NULL);
	if (0 != ret) {
		fprintf(stderr, "pthread_create[%s:%d].\n", __FILE__, __LINE__);
		return -1;
	}
	
	return 0;
}

void *aicore_test_thread(void *param)
{
	int counter = 0;
	double duration = 0;
	struct timeval t1, t2;
	float threshold = 0.4f;
	const struct timespec req = {0, 1000};
	gettimeofday(&t1, NULL);
	
	while (!quit) {
		object_t object;
		size_t num = ai_core_fetch_object(&object, 1, threshold);
		if (num <= 0) {
			nanosleep(&req, NULL);
			continue;
		}

		gettimeofday(&t2, NULL);
		duration = ((double)t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;
		++counter;
		printf("detected object[%d,%d,%d,%d], frame rate %ffps.\n", object.x, object.y, object.w, object.h, counter / duration);
	}
	
	return (void *)(0);
}

void draw_bounding_box(bitmap *bmp, int x, int y, int w, int h)
{
	const int pitch = get_bmp_pitch(bmp);
	const int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	const int nchannels = bit_count >> 3;
	const int color[3] = {0, 255, 255};
	const int left = x;
	const int right = x + w - 1;
	const int top = y;
	const int bottom = y + h - 1;
	
	for (int c = 0; c < nchannels; ++c) {
		for (int y = top; y < bottom; ++y) {
			data[y * pitch + left * nchannels + c] = color[c];
			data[y * pitch + right * nchannels + c] = color[c];
		}
		
		for (int x = left; x < right; ++x) {
			data[top * pitch + x * nchannels + c] = color[c];
			data[bottom * pitch + x * nchannels + c] = color[c];
		}
	}
}

void wait_for_thread_dead(pthread_t tid)
{
	int timer = 1000;
	struct timespec req = {0, 10000000};
	while (timer--) {
		int ret = pthread_kill(tid, 0);
		if (ESRCH == ret) {
			fprintf(stderr, "the thread didn't exists or already quit[%s:%d].\n", __FILE__, __LINE__);
			return;
		} else if (EINVAL == ret) {
			fprintf(stderr, "signal is invalid[%s:%d].\n", __FILE__, __LINE__);
			return;
		} else {
			nanosleep(&req, NULL);
			continue;
		}
	}
}

#ifdef __linux__
int kbhit(void)
{
	struct termios oldt, newt;
	tcgetattr(STDIN_FILENO, &oldt);
	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);
	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	
	int oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
	
	int ch = getchar();
	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF) {
		ungetc(ch, stdin);
		return 1;
	}
	
	return 0;
}
#endif