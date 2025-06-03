#include "yolov5s.h"
#include "post_process.h"


// 静态函数，用于打印 rknn_tensor_attr 结构体的信息
static void print_tensor_attr(rknn_tensor_attr *attr)
{
    // 构建形状字符串，例如 "640,480,3" 表示一个 640x480 的 RGB 图像
    string shape_str = attr->n_dims < 1 ? "" : to_string(attr->dims[0]);
    for(int i = 1; i < attr->n_dims; i++)
    {
        string current_str = to_string(attr->dims[i]);
        shape_str += "," + current_str;
    }

    // // 打印张量的索引、名称、维度数、维度、大小和格式
    // printf("index = %d, name = %s， n_dims = %d, dims = [%s], \nsize = %d, fmt = %s\n", 
    //         attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->size, get_format_string(attr->fmt));
    // printf("\n");
}

Yolov5s::Yolov5s(const char* model_path, int npu_index)
{
    int ret; 
    model_data = load_model(model_path, this->model_size);
    /* 模型初始化加载到RKNN中 */
    ret = rknn_init(&this->context, model_data, this->model_size , RKNN_FLAG_PRIOR_HIGH, NULL);
    if (ret != 0)
    {  
        printf("rknn init failed! error code: %d\n", ret);
    } 
    else 
    {
        printf("yolo %d初始化成功！\n",npu_index);
    }

    /* 对不同线程分配NPU，加速计算 */
    if(npu_index %4 == 0)     {   ret = rknn_set_core_mask(this->context,RKNN_NPU_CORE_0);}
    else if(npu_index %4 == 1){   ret = rknn_set_core_mask(this->context,RKNN_NPU_CORE_1);}
    else                     {   ret = rknn_set_core_mask(this->context,RKNN_NPU_CORE_2);}
    if (ret != 0){      printf("npu set failed! error code: %d\n", ret);} 

    /* 够查询获取到模型输入输出信息、逐层运行时间、模型推理的总时间、
        SDK版本、内存占用信息、用户自定义字符串等信息 */
    ret = rknn_query(context, RKNN_QUERY_IN_OUT_NUM,&this->num_tensors ,sizeof(this->num_tensors) );
    if (ret != 0){      printf("rknn_query failed! error code: %d\n", ret);    } 
    // printf("输入 tensor个数为：%d \n",num_tensors.n_output);
    // printf("输出 tensor个数为：%d \n",num_tensors.n_input);

    /* 根据tensor信息调整输入和输出的tensor个数 */
    input_attrs.resize(num_tensors.n_input);
    output_attrs.resize(num_tensors.n_output);
    
    /* 获取模型需要的输入和输出的tensor信息 */
    for(int i = 0;i < num_tensors.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(context, RKNN_QUERY_INPUT_ATTR,&(this->input_attrs[i]) ,sizeof( this->input_attrs[i]) );
        if (ret != 0)   {printf("rknn_query input_attrs failed! error code: %d\n", ret);} 
        printf("输入  的tensor%d  属性为：\n",i);
        print_tensor_attr(&(this->input_attrs[i]));
    }

    for(int i = 0;i < num_tensors.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(context, RKNN_QUERY_OUTPUT_ATTR,&(this->output_attrs[i]) ,sizeof( this->output_attrs[i]) );
        if (ret != 0)   {printf("rknn_query output_attrs failed! error code: %d\n", ret);} 
        // printf("输出  的tensor%d  属性为：\n",i);
        print_tensor_attr(&(this->output_attrs[i]));
    }

    /* 获取模型要求输入图像的参数信息 */
    // 根据输入张量的格式确定模型的维度信息
    if(input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        model_channel = input_attrs[0].dims[1];
        model_height = input_attrs[0].dims[2];
        model_width = input_attrs[0].dims[3];
    }
    
    else if(input_attrs[0].fmt == RKNN_TENSOR_NHWC)
    {
        model_height = input_attrs[0].dims[1];
        model_width = input_attrs[0].dims[2];
        model_channel = input_attrs[0].dims[3];
    }
}

// 析构函数中
Yolov5s::~Yolov5s()
{
    if (context) {
        rknn_destroy(context); // 释放RKNN上下文
    }
    free(this->model_data);
}


unsigned char * Yolov5s::load_model(const char* model_path, unsigned int &model_size)
{
    FILE *fp;
    unsigned char* model_data;

    fp = fopen(model_path, "rb");
    if(fp == NULL)
    {
        printf("open model failed!\n");
    }

    int ret = fseek(fp, 0, SEEK_END);
    if(ret)
    {
        printf("fseek err : %d\n",ret);
    }

    model_size = ftell(fp);
    
    model_data =(unsigned char *) malloc(model_size);

    ret = fseek(fp, 0, SEEK_SET);
    if(ret)
    {
        printf("fseek err : %d\n",ret);
    }

    ret = fread(model_data, 1, model_size, fp);
    if(ret<0)
    {
        printf("read model failed! err: %d\n",ret);
    }
    
    return model_data;
}

int Yolov5s::inference_image(const Mat& orig_img, detect_result_group_t &result_group)
{
     int ret = 0;

    float nms_threshold       = NMS_THRESHOLD;
    float box_conf_threshold  = BOX_THRESHOLD;

    Mat bkg;
    this->img_height = orig_img.rows; // 获取原始图像的高度
    this->img_width = orig_img.cols; // 获取原始图像的宽度
    this->img_channel = orig_img.channels(); // 获取原始图像的通道数

    // 检查图像尺寸是否为16的倍数，如果不是则进行填充
    if(img_width % 16 != 0 || img_height % 16 != 0)
    {
        int bkg_width = (img_width + 15) / 16 * 16;
        int bkg_height = (img_height + 15) / 16 * 16;

        bkg = Mat(bkg_height, bkg_width, CV_8UC3, cv::Scalar(0, 0, 0)); // 创建背景图像
        orig_img.copyTo(bkg(cv::Rect(0, 0, orig_img.cols, orig_img.rows))); // 将原始图像复制到背景图像中
        cv::imwrite("img_bkg.jpg", bkg); // 保存背景图像
        this->img_width = bkg_width; // 更新图像宽度
        this->img_height = bkg_height; // 更新图像高度
    }
    else
    {
        // 尺寸已经是 16 的倍数，不需要 padding
        bkg = orig_img.clone(); // 创建 bkg，并复制原始图像内容 (深拷贝)
        // 或者，如果你只是想避免 bkg 为空，也可以使用浅拷贝，但深拷贝更安全
        // bkg = orig_img; // 浅拷贝，bkg 和 orig_img 共享数据，不推荐，可能引起意外修改
    }
    //error
    // this->img_height    = (orig_img.rows + 15) /16 * 16;
    // this->img_width     = (orig_img.cols + 15) /16 * 16;
    // this->img_channel   = orig_img.channels();

    int resize_height   = this->model_height;
    int resize_width    = this->model_width;
    int resize_channel  = this->model_channel;

// 打印图像的原始尺寸和调整后的尺寸
    // printf("Image Height: %d\n", img_height);
    // printf("Image Width: %d\n", img_width);
    // printf("Image Channels: %d\n", img_channel);

    // printf("Resize Height: %d\n", resize_height);
    // printf("Resize Width: %d\n", resize_width);
    // printf("Resize Channels: %d\n", resize_channel);

 // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // //opencv
    // Mat img_cvt;
    // Mat img_resize;
    // start       = std::chrono::high_resolution_clock::now();
    // cv::cvtColor(orig_img, img_cvt, cv::COLOR_BGR2RGB);
    // cv::resize(img_cvt, img_resize, Size(resize_height, resize_width), 0, 0, cv::INTER_LINEAR);
    // end         = std::chrono::high_resolution_clock::now();
    // duration    =std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // printf("opencv process time : %ld ms.\n", duration.count());
    // cv::imwrite("img_cv_linear.jpg", img_resize);    

    // rga进行图像处理
    Mat img_rga;
    Mat img_cvt;
    start      = std::chrono::high_resolution_clock::now();
    char *src_buf, *dst_buf, *src_cvt_buf;
    rga_buffer_handle_t src_handle, dst_handle, src_cvt_handle;

     // 分配内存
    src_buf = (char *)malloc(img_height * img_width * img_channel);
    src_cvt_buf = (char *)malloc(img_height * img_width * img_channel);
    dst_buf = (char *)malloc(resize_height * resize_width * resize_channel);

    if (bkg.empty()) 
    {
        printf("错误：bkg Mat 对象为空，可能未正确初始化！\n");
        // 可以选择直接返回错误，或者采取其他错误处理措施
        return -1; // 返回错误代码，表示初始化失败
    }
    // 复制数据并初始化内存
    memcpy(src_buf, bkg.data, img_height * img_width * img_channel);
    memset(src_cvt_buf, 0x00, img_height * img_width * img_channel);
    memset(dst_buf, 0x00, resize_height * resize_width * resize_channel);

    // 导入缓冲区
    src_handle = importbuffer_virtualaddr(src_buf, img_height * img_width * img_channel);
    src_cvt_handle = importbuffer_virtualaddr(src_cvt_buf, img_height * img_width * img_channel);
    dst_handle = importbuffer_virtualaddr(dst_buf, resize_height * resize_width * resize_channel);

    if(src_handle == 0 || src_cvt_handle == 0|| dst_handle == 0)
    {
        printf("import va failed.\n");
    }
    
    // 定义rga缓冲区
    rga_buffer_t src = wrapbuffer_handle(src_handle, img_width,img_height, RK_FORMAT_BGR_888);
    rga_buffer_t src_cvt = wrapbuffer_handle(src_cvt_handle, img_width,img_height, RK_FORMAT_RGB_888);
    rga_buffer_t dst = wrapbuffer_handle(dst_handle, resize_width, resize_height, RK_FORMAT_RGB_888);

    // 检查图像格式
    ret = imcheck(src, dst, {}, {});
    if(ret != IM_STATUS_NOERROR)
    {
        printf("%d, imcheck error! %s\n", __LINE__,  imStrError((IM_STATUS)ret));
        ret = -1;
        // goto release_buffer;
    }

    // 检查图像格式
    ret = imcvtcolor(src, src_cvt, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
    if(ret == IM_STATUS_SUCCESS)
    {
        // printf("convert color OK!\n");
    }   
    else
    {
        printf("%d, cvtColor error! %s\n", __LINE__,  imStrError((IM_STATUS)ret));
        ret = -1;
        // goto release_buffer;
    }

    // 调整图像大小
    ret = imresize(src_cvt, dst);
    if(ret == IM_STATUS_SUCCESS)
    {
        // printf("resize color OK!\n");
    }
    else
    {
        printf("%d, resize error! %s\n", __LINE__,  imStrError((IM_STATUS)ret));
        ret = -1;
        // goto release_buffer;
    }
    
    // 记录结束时间并计算处理时间
    end         = std::chrono::high_resolution_clock::now();
    duration    =std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // printf("rga process time : %ld ms.\n", duration.count());
    // 记录结束时间并计算处理时间
    img_cvt = Mat(img_height, img_width, CV_8UC3, src_cvt_buf);
    cv::imwrite("img_rga_cvt.jpg", img_cvt);
    img_rga = Mat(resize_height, resize_width, CV_8UC3, dst_buf);
    cv::imwrite("img_rga_rsz.jpg", img_rga);

     // 推理
    start = std::chrono::high_resolution_clock::now();
    //////printf("set inputs...\n");
    int inputs_num = num_tensors.n_input;
    rknn_input inputs[inputs_num];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = model_height * model_width * model_channel;
    inputs[0].pass_through = false;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = dst_buf;

    // 设置模型输入
    rknn_inputs_set(context, inputs_num, inputs);

       ////printf("set outputs");
    int outputs_num = num_tensors.n_output;
    rknn_output outputs[outputs_num];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < outputs_num; i++)
    {
        outputs[i].want_float = 0;
    }
    ////printf("model inferencing...\n");
    ret = rknn_run(context, NULL);
    if (ret == 0)
    {
        //printf("model inferencing OK!\n");
    }

    // 获取模型输出
    rknn_outputs_get(context, outputs_num, outputs, NULL);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //printf("model inferencing time : %ld ms.\n", duration.count());

    // postprocess  640 / 960
    float scale_w = (float)model_width / img_width;
    float scale_h = (float)model_height / img_height;

    vector<int32_t> qnt_zps;
    vector<float> qnt_scales;

    for (int i = 0; i < outputs_num; i++)
    {
        //printf("第%d个output的zp和scale分别是：",i);
        qnt_zps.emplace_back(output_attrs[i].zp);
        qnt_scales.emplace_back(output_attrs[i].scale);
        // printf("%d,%f\n",output_attrs[i].zp,output_attrs[i].scale);
    }

    //进行后处理操作
    int debug = post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, 
                model_height, model_width, box_conf_threshold, nms_threshold,
                 scale_w, scale_h, qnt_zps, qnt_scales,result_group);


    draw_result(orig_img,result_group);


    ret = 0;
// release_buffer:
    //free
    // 释放资源
    if(src_handle)
    {
        releasebuffer_handle(src_handle);
    }
    if(src_cvt_handle)
    {
        releasebuffer_handle(src_cvt_handle);
    }
    if(dst_handle)
    
    {
        releasebuffer_handle(dst_handle);
    }
    free(src_buf);
    free(dst_buf);
    free(src_cvt_buf);

    return ret;
}
 
int Yolov5s::draw_result(const cv::Mat &orig_img, detect_result_group_t& result_group)
{
    for(int i = 0; i < result_group.box_count; i++)
    {
        int xmin = result_group.result[i].box.xmin;
        int ymin = result_group.result[i].box.ymin;
        int xmax = result_group.result[i].box.xmax;
        int ymax = result_group.result[i].box.ymax;

        cv::rectangle(orig_img, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 0, 255), 3);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) // 设置固定小数点表示法，保留两位小数
            << result_group.result[i].label << ":"
            << result_group.result[i].box_conf*100 << " %";
        std::string img_label = ss.str();
        
        cv::putText(
                        orig_img,                    // 要添加文字的图像
                        img_label,                   // 要添加的文字内容
                        cv::Point(xmin, ymin-15),       // 文字左下角的坐标 (x, y)
                        FONT_HERSHEY_SIMPLEX,        // 字体类型 (见下文)
                        0.8,                           // 字体缩放比例 (相对于原始字体大小)
                        cv::Scalar(0, 0, 255),            // 文字颜色 (BGR 格式)
                        1,           // 文字线条粗细
                        cv::LINE_8,   // 线条类型 (见下文)
                        false // 如果为 true，则原点在左上角
                    );
        
    }
    
    return 0;
}
