#include "stdafx.h"  
#include "sharedmatting.h"
#include <string>

using namespace std;

int main()
{
	char fileAddr[64] = { 0 };

	for (int n = 1; n < 28; ++n) {
		SharedMatting sm;

		//sprintf(fileAddr, "C:/Users/Chuangkit_Developer7/Desktop/input.png", n / 10, n % 10);
		sm.loadImage("C:/Users/Chuangkit_Developer7/Desktop/1-input.png");

		//sprintf(fileAddr, "C:/Users/Chuangkit_Developer7/Desktop/trimap.png", n / 10, n % 10);
		sm.loadTrimap("C:/Users/Chuangkit_Developer7/Desktop/1-trimap.png");

		sm.solveAlpha();

		//sprintf(fileAddr, "C:/Users/Chuangkit_Developer7/Desktop/result.png", n / 10, n % 10);
		sm.save("C:/Users/Chuangkit_Developer7/Desktop/1-result.png");
	}

	return 0;
}