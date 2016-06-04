#include <vector>
#include <cmath>

#include "caffe/extra_layers/spatial_transform_layer.hpp"

namespace caffe {

template <typename Dtype>
void SpatialTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[1]->shape(1)*bottom[1]->shape(2), 6)<<"Theta should be 6-dimension!";
    
    num = bottom[0]->shape(0);
    channel = bottom[0]->shape(1);
    height = bottom[0]->shape(2);
    width = bottom[0]->shape(3);

    output_height = height;
    output_width = width;

    //The number of integer points in the grid;
    point_cnt = output_height * output_width;
    x_cnt = output_height;
    y_cnt = output_width;

    //Init output_grid.
    vector<int> output_grid_shape(2);
    output_grid_shape[0] = point_cnt;
    output_grid_shape[1] = 3;
    output_grid->Reshape(output_grid_shape);
    Dtype* output_grid_data = output_grid->mutable_cpu_data();
    for(int i = 0; i < point_cnt; i++){
        output_grid_data[3*i] = (i / output_width) * 1.0 / output_height * 2 -1;
        output_grid_data[3*i+1] = (i % output_height) * 1.0 / output_height * 2 - 1;
 	output_grid_data[3*i+2] = 1;
    }
    
    //Init input_grid.
    vector<int> input_grid_shape(3);
    input_grid_shape[0] = bottom[0]->shape(0);
    input_grid_shape[1] = point_cnt;
    input_grid_shape[2] = 2;
    input_grid->Reshape(input_grid_shape);
    
}

template <typename Dtype>
void SpatialTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
    vector<int> top_shape(4);
    top_shape[0] = bottom[0]->shape(0);
    top_shape[1] = bottom[0]->shape(1);
    top_shape[2] = output_height;
    top_shape[3] = output_width;
    top[0]->Reshape(top_shape);

}

template <typename Dtype>
void SpatialTransformLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
    const Dtype* U = bottom[0]->cpu_data();
    const Dtype* theta = bottom[1]->cpu_data();
    const Dtype* output_grid_coor = output_grid->cpu_data(); 
    Dtype* input_grid_data = input_grid->mutable_cpu_data();    
    Dtype* V = top[0]->mutable_cpu_data();  
    
    caffe_set(input_grid->count(), (Dtype)0, input_grid_data);
    caffe_set(top[0]->count(), (Dtype)0, V);

    int index;
    Dtype x, y, xx, yy;
    for(int n = 0; n < num; n++){
        Dtype* coordinates = input_grid_data + n*point_cnt*2;
        caffe_cpu_gemm(CblasNoTrans, CblasTrans, point_cnt, 2, 3, (Dtype)1, output_grid_coor, theta+6*n, (Dtype)0, coordinates);
        for(int c = 0; c < channel; c++){
            for(int h = 0; h < output_height; h++){
  	        for(int w = 0; w < output_width; w++){
		    index = h * output_width + w;
		    x = coordinates[2*index];
                    xx = (x+1) / 2 * height;
                    y = coordinates[2*index + 1];
                    yy = (y+1) / 2 * width;
		    for(int i = floor(xx); i <= ceil(xx); i++){
			for(int j = floor(yy); j <= ceil(yy); j++){
			    V[top[0]->offset(n, c, h, w)] += max(0, 1-abs(xx-i)) * max(0, 1-abs(yy-j)) * U[bottom[0]->offset(n, c, i, j)]; 
			}
		    }	    
		}
	    }
	}       
                       
    }
   
     
}

template <typename Dtype>
void SpatialTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   
    const Dtype* diff_V = top[0]->cpu_diff();
    const Dtype* input_grid_data = input_grid->cpu_data();
    const Dtype* U = bottom[0]->cpu_data();
    Dtype* diff_U = bottom[0]->mutable_cpu_diff();
    Dtype* diff_theta = bottom[1]->mutable_cpu_diff();
    Dtype* diff_input_grid =  input_grid->mutable_cpu_diff();
    
    caffe_set(bottom[0]->count(), (Dtype)0, diff_U);
    caffe_set(bottom[1]->count(), (Dtype)0, diff_theta);
    caffe_set(input_grid->count(), (Dtype)0, diff_input_grid);

    int index;
    Dtype x, y, xx, yy, xs_diff, ys_diff, condition_x, condition_y;
    for(int n = 0; n < num; n++){
        const Dtype* coordinates = input_grid_data + 2*n*point_cnt;
        Dtype* coordinates_diff = diff_input_grid + 2*n*point_cnt;
        for(int h = 0; h < output_height; h++){
	    for(int w = 0; w < output_width; w++){
         	index = h * output_width + w;
		x = coordinates[2*index];
		y = coordinates[2*index+1];
                xx = (x+1)/2*height;
		yy = (y+1)/2*width;
		xs_diff = 0;
		ys_diff = 0;
                for(int c = 0; c < channel; c++){
		    for(int i = floor(xx); i < ceil(xx); i++){
			for(int j = floor(yy); j < ceil(yy); j++){
			    w = max(0, 1-abs(xx-i)) * max(0, 1-abs(yy-j));
			    diff_U[bottom[0]->offset(n, c, h, w)] += w * diff_V[top[0]->offset(n, c, h, w)];
			    condition_x = 0;
			    condition_y = 0;
			    if(abs(xx-i) < 1){
				if(xx >= i){
				    condition_x = 1;
				}else{
				    condition_x = -1;
				}

			    }
			    if(abs(yy-j) < 1){
				if(yy >= j){
				    condition_y = 1;
				}else{
				    condition_y = -1;
				}
			    }
			    xs_diff += condition_x * max(0, 1-abs(yy-j)) * U[bottom[0]->offset(n, c, h, w)] * diff_V[top[0]->offset(n, c, h, w)] * height / 2;
			    ys_diff += condition_y * max(0, 1-abs(xx-i)) * U[bottom[0]->offset(n, c, h, w)] * diff_V[top[0]->offset(n, c, h, w)] * width / 2;

			}
		    }
		    coordinates_diff[2*index] += xs_diff;
		    coordinates_diff[2*index+1] += ys_diff; 
		}
		xs_diff = coordinates_diff[2*index];
		ys_diff = coordinates_diff[2*index+1];
                diff_theta[6*n] = xs_diff * (h * 1.0 / output_height * 2 - 1);
		diff_theta[6*n+1] = xs_diff * (w * 1.0 / output_width * 2 - 1);
		diff_theta[6*n+2] = xs_diff;
		diff_theta[6*n+3] = ys_diff * (h * 1.0 / output_height * 2 - 1);
		diff_theta[6*n+4] = ys_diff * (w * 1.0 / output_width * 2 - 1);
		diff_theta[6*n+5] = ys_diff;
                   	        
	    }
        }

    }
}

#ifdef CPU_ONLY
STUB_GPU(SpatialTransformLayer);
#endif

INSTANTIATE_CLASS(SpatialTransformLayer);
REGISTER_LAYER_CLASS(SpatialTransform);

}  // namespace caffe
