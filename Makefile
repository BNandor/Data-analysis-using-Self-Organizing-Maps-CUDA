ocr: digits.cpp ocr.cpp ocr.h
	g++ digits.cpp ocr.cpp  ocr.h  -Dcrossvalidateregression --std=c++11  -lpthread -larmadillo -O3 
#g++ digits.cpp ocr.cpp  ocr.h -Dknn -Dcentroid -Dlregression    --std=c++11  -lpthread -larmadillo -O3 
