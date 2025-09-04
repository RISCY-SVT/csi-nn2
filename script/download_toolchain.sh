#! /bin/bash

ROOT_PATH=$PWD
GCC_LINK="https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1755679709552/Xuantie-900-gcc-linux-6.6.0-glibc-x86_64-V3.2.0-20250627.tar.gz"
TAR_FILE=$ROOT_PATH/tools/Xuantie-900-gcc-linux-toolchain.tar.gz

# create dir
GCC_PATH=tools/gcc-toolchain
mkdir -p $GCC_PATH

# download XuanTie-GNU-toolchain
if [ ! -f "$TAR_FILE" ]; then
	wget -O $TAR_FILE $GCC_LINK
fi

# unzip file
if [ -f "$TAR_FILE" ]; then
	if [ ! -d $ROOT_PATH/$GCC_PATH/bin ]; then
		tar -zxvf $TAR_FILE -C $ROOT_PATH/$GCC_PATH --strip-components 1
	fi
	rm -rf $TAR_FILE
fi

echo "Download toolchain in $ROOT_PATH/tools/"
echo "Set env variable:
	export PATH=$ROOT_PATH/$GCC_PATH/bin:$ \bPATH"
