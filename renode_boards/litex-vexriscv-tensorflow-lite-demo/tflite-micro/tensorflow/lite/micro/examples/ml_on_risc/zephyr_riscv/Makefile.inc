ifeq ($(TARGET), zephyr_vexriscv)
	export ZEPHYR_TOOLCHAIN_VARIANT?=zephyr
	export TOOLCHAIN_BASE=${ZEPHYR_SDK_INSTALL_DIR}/riscv64-zephyr-elf/riscv64-zephyr-elf
	export TOOLCHAIN_VERSION=9.2.0
	PROJECT_INCLUDES += ${CURDIR} ${TOOLCHAIN_BASE}/include/c++/${TOOLCHAIN_VERSION} ${TOOLCHAIN_BASE}/include ${TOOLCHAIN_BASE}/include/c++/${TOOLCHAIN_VERSION}/riscv64-zephyr-elf/rv32i/ilp32
	ZEPHYR_ML_ON_RISC_SRCS = \
tensorflow/lite/micro/examples/ml_on_risc/zephyr_riscv/src/assert.cc \
tensorflow/lite/micro/examples/ml_on_risc/main.cc \
tensorflow/lite/micro/examples/ml_on_risc/dequantize_output.cc \
tensorflow/lite/micro/examples/ml_on_risc/extract_codebook_index.cc \
tensorflow/lite/micro/examples/ml_on_risc/main_functions.cc \
tensorflow/lite/micro/examples/ml_on_risc/quantize_input.cc \
tensorflow/lite/micro/examples/ml_on_risc/read_sample_from_file_local_v2.cc \
tensorflow/lite/micro/examples/ml_on_risc/c_models/model_py_test_seed0_grid1200_M3232_Mbar8_10000_60_in1024_out1024_nl3_hul1024_4096_4096_model_data.cc \
boards/litex_vexriscv.overlay \
prj.conf

$(eval $(call generate_project,cmake,zephyr_cmake_project.cmake,ml_on_risc,$(MICROLITE_CC_SRCS) $(THIRD_PARTY_CC_SRCS) $(ZEPHYR_ML_ON_RISC_SRCS) $(MICROLITE_CC_HDRS) $(THIRD_PARTY_CC_HDRS) $(ml_on_risc_HDRS),,$(LDFLAGS) $(MICROLITE_LIBS),$(CXXFLAGS),$(CCFLAGS),))

$(PRJDIR)ml_on_risc/cmake/CMakeLists.txt: $(PRJDIR)ml_on_risc/cmake/zephyr_cmake_project.cmake
	@sed -E 's#\%\{INCLUDE_DIRS\}\%#$(PROJECT_INCLUDES)#g' $< > $@

#We are skipping here copy of `zephyr` third_party repository 
#To compile standalone project ZEPHYR_BASE enviroment variable should be set
ml_on_risc_bin: generate_ml_on_risc_cmake_project $(PRJDIR)ml_on_risc/cmake/CMakeLists.txt
	( \
	  . ${ZEPHYR_BASE}/venv-zephyr/bin/activate; \
	  cmake -B${GENDIR}ml_on_risc/build -DBOARD="litex_vexriscv" -H${PRJDIR}ml_on_risc/cmake/ -DPython_ROOT_DIR=${ZEPHYR_BASE}/venv-zephyr/bin/; \
	  make -C ${GENDIR}ml_on_risc/build; \
	)
endif
