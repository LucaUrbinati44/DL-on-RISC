ifeq ($(TARGET), sparkfun_edge)
  INCLUDES += \
    -I$(APOLLO3_SDK)/$(SF_BSPS_DEST)/common/third_party/lis2dh12/

  THIRD_PARTY_CC_SRCS += \
    $(APOLLO3_SDK)/$(SF_BSPS_DEST)/common/third_party/lis2dh12/lis2dh12_platform_apollo3.c \
    $(APOLLO3_SDK)/boards_sfe/common/third_party/lis2dh12/lis2dh12_reg.c

  THIRD_PARTY_CC_HDRS += \
    $(APOLLO3_SDK)/boards_sfe/common/third_party/lis2dh12/lis2dh12_platform_apollo3.h \
    $(APOLLO3_SDK)/boards_sfe/common/third_party/lis2dh12/lis2dh12_reg.h
endif

ml_on_risc_SRCS := \
tensorflow/lite/micro/examples/ml_on_risc/main.cc \
tensorflow/lite/micro/examples/ml_on_risc/dequantize_output.cc \
tensorflow/lite/micro/examples/ml_on_risc/extract_codebook_index.cc \
tensorflow/lite/micro/examples/ml_on_risc/main_functions.cc \
tensorflow/lite/micro/examples/ml_on_risc/quantize_input.cc \
tensorflow/lite/micro/examples/ml_on_risc/read_sample_from_file.cc \
tensorflow/lite/micro/examples/ml_on_risc/c_models/model_py_test_seed0_grid1200_M3232_Mbar8_10000_60_in1024_out1024_nl3_hul1024_4096_4096_model_data.cc
#tensorflow/lite/micro/examples/ml_on_risc/write_sample_to_file.cc \

ml_on_risc_local_SRCS := \
tensorflow/lite/micro/examples/ml_on_risc/main.cc \
tensorflow/lite/micro/examples/ml_on_risc/dequantize_output.cc \
tensorflow/lite/micro/examples/ml_on_risc/extract_codebook_index.cc \
tensorflow/lite/micro/examples/ml_on_risc/main_functions_local.cc \
tensorflow/lite/micro/examples/ml_on_risc/quantize_input.cc \
tensorflow/lite/micro/examples/ml_on_risc/read_sample_from_file_local_v2.cc \
tensorflow/lite/micro/examples/ml_on_risc/write_sample_to_file_local.cc \
tensorflow/lite/micro/examples/ml_on_risc/c_models/model_py_test_seed0_grid1200_M3232_Mbar8_10000_60_in1024_out1024_nl3_hul1024_4096_4096_model_data.cc

ml_on_risc_HDRS := \
tensorflow/lite/micro/examples/ml_on_risc/config.h \
tensorflow/lite/micro/examples/ml_on_risc/dequantize_output.h \
tensorflow/lite/micro/examples/ml_on_risc/extract_codebook_index.h \
tensorflow/lite/micro/examples/ml_on_risc/main_functions.h \
tensorflow/lite/micro/examples/ml_on_risc/quantize_input.h \
tensorflow/lite/micro/examples/ml_on_risc/read_sample_from_file.h \
tensorflow/lite/micro/examples/ml_on_risc/c_models/model_py_test_seed0_grid1200_M3232_Mbar8_10000_60_in1024_out1024_nl3_hul1024_4096_4096_model_data.h
#tensorflow/lite/micro/examples/ml_on_risc/write_sample_to_file.h \

ml_on_risc_local_HDRS := \
tensorflow/lite/micro/examples/ml_on_risc/config.h \
tensorflow/lite/micro/examples/ml_on_risc/dequantize_output.h \
tensorflow/lite/micro/examples/ml_on_risc/extract_codebook_index.h \
tensorflow/lite/micro/examples/ml_on_risc/main_functions_local.h \
tensorflow/lite/micro/examples/ml_on_risc/quantize_input.h \
../renode/test_set_small_10.h \
tensorflow/lite/micro/examples/ml_on_risc/read_sample_from_file_local_v2.h \
tensorflow/lite/micro/examples/ml_on_risc/write_sample_to_file_local.h \
tensorflow/lite/micro/examples/ml_on_risc/c_models/model_py_test_seed0_grid1200_M3232_Mbar8_10000_60_in1024_out1024_nl3_hul1024_4096_4096_model_data.h

#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/ml_on_risc/*/Makefile.inc)

# Builds a standalone binary
$(eval $(call microlite_test,ml_on_risc,\
$(ml_on_risc_SRCS),$(ml_on_risc_HDRS)))

# LUCA
# Builds a standalone binary for local file input
$(eval $(call microlite_test,ml_on_risc_local,\
$(ml_on_risc_local_SRCS),$(ml_on_risc_local_HDRS)))