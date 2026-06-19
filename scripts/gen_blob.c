#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include "nnet.h"
#include "rnnoise_data.c"

/* Arrays the FLOAT compute path actually consumes (see init_rnnoise + compute_linear). */
static const char *keep[] = {
  "conv1_weights_float","conv1_bias",
  "conv2_weights_float","conv2_bias",
  "gru1_input_weights_float","gru1_input_weights_idx","gru1_input_bias",
  "gru1_recurrent_weights_float","gru1_recurrent_weights_idx","gru1_recurrent_weights_diag","gru1_recurrent_bias",
  "gru2_input_weights_float","gru2_input_weights_idx","gru2_input_bias",
  "gru2_recurrent_weights_float","gru2_recurrent_weights_idx","gru2_recurrent_weights_diag","gru2_recurrent_bias",
  "gru3_input_weights_float","gru3_input_weights_idx","gru3_input_bias",
  "gru3_recurrent_weights_float","gru3_recurrent_weights_idx","gru3_recurrent_weights_diag","gru3_recurrent_bias",
  "dense_out_weights_float","dense_out_bias",
  "vad_dense_weights_float","vad_dense_bias",
  NULL
};
static int wanted(const char *n){ for(int i=0;keep[i];i++) if(!strcmp(keep[i],n)) return 1; return 0; }

int main(void){
  FILE *fout = fopen("rnnoise_default.bin","wb");
  unsigned char zeros[WEIGHT_BLOCK_SIZE] = {0};
  long total=0; int n=0;
  for (int i=0; rnnoise_arrays[i].name != NULL; i++){
    if(!wanted(rnnoise_arrays[i].name)) continue;
    WeightHead h;
    memcpy(h.head,"DNNw",4);
    h.version = WEIGHT_BLOB_VERSION;
    h.type = rnnoise_arrays[i].type;
    h.size = rnnoise_arrays[i].size;
    h.block_size = (h.size+WEIGHT_BLOCK_SIZE-1)/WEIGHT_BLOCK_SIZE*WEIGHT_BLOCK_SIZE;
    memset(h.name,0,sizeof(h.name));
    strncpy(h.name, rnnoise_arrays[i].name, sizeof(h.name)-1);
    fwrite(&h,1,WEIGHT_BLOCK_SIZE,fout);
    fwrite(rnnoise_arrays[i].data,1,h.size,fout);
    fwrite(zeros,1,h.block_size-h.size,fout);
    total += WEIGHT_BLOCK_SIZE + h.block_size;
    printf("  %-34s type=%d size=%d\n", rnnoise_arrays[i].name, h.type, h.size);
    n++;
  }
  fclose(fout);
  printf("wrote %d arrays, %ld bytes\n", n, total);
  return 0;
}
