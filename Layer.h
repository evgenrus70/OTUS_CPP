#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;

template <typename T>
class vector_4d
{
    public:
        typedef typename std::vector<T>::size_type size_type;
        typedef typename std::vector<T>::iterator iterator;
        typedef typename std::vector<T>::const_iterator const_iterator;

        vector_4d (){}

        vector_4d(size_type x, size_type y, size_type z, size_type w)
        :x_(x)
        ,y_(y)
        ,z_(z)
        ,w_(w)
        ,data_(x_*y_*z_*w_)
        {}

        const T& operator()(size_type x_pos, size_type y_pos, size_type z_pos, size_type w_pos) const
        {
            size_type pos_in_4d_plane = x_ * y_ * z_ * w_pos;
            size_type pos_in_3d_plane = x_ * y_ * z_pos;
            size_type pos_in_2d_plane = x_ * y_pos;
            size_type pos_in_1d_plane = x_pos;
            return data_[pos_in_4d_plane + pos_in_3d_plane + pos_in_2d_plane + pos_in_1d_plane];
        }

        T& operator()(size_type x_pos, size_type y_pos, size_type z_pos, size_type w_pos)
        {
            size_type pos_in_4d_plane = x_ * y_ * z_ * w_pos;
            size_type pos_in_3d_plane = x_ * y_ * z_pos;
            size_type pos_in_2d_plane = x_ * y_pos;
            size_type pos_in_1d_plane = x_pos;
            return data_[pos_in_4d_plane + pos_in_3d_plane + pos_in_2d_plane + pos_in_1d_plane];
        }

        iterator begin() { return data_.begin(); }
        iterator end()   { return data_.end(); }

        const_iterator begin() const { return data_.begin(); }
        const_iterator end()   const { return data_.end(); }

    private:
        size_type x_;
        size_type y_;
        size_type z_;
        size_type w_;

        std::vector<T> data_;
};

template <typename T>
class vector_3d
{
    public:
        typedef typename std::vector<T>::size_type size_type;
        typedef typename std::vector<T>::iterator iterator;
        typedef typename std::vector<T>::const_iterator const_iterator;

        vector_3d (){}

        vector_3d(size_type x, size_type y, size_type z)
        :x_(x)
        ,y_(y)
        ,z_(z)
        ,data_(x_*y_*z_)
        {}

        const T& operator()(size_type x_pos, size_type y_pos, size_type z_pos) const
        {
            size_type pos_in_3d_plane = x_ * y_ * z_pos;
            size_type pos_in_2d_plane = x_ * y_pos;
            size_type pos_in_1d_plane = x_pos;
            return data_[pos_in_3d_plane + pos_in_2d_plane + pos_in_1d_plane];
        }

        T& operator()(size_type x_pos, size_type y_pos, size_type z_pos)
        {
            size_type pos_in_3d_plane = x_ * y_ * z_pos;
            size_type pos_in_2d_plane = x_ * y_pos;
            size_type pos_in_1d_plane = x_pos;
            return data_[pos_in_3d_plane + pos_in_2d_plane + pos_in_1d_plane];
        }

        iterator begin() { return data_.begin(); }
        iterator end()   { return data_.end(); }

        const_iterator begin() const { return data_.begin(); }
        const_iterator end()   const { return data_.end(); }

    private:
        size_type x_;
        size_type y_;
        size_type z_;

        std::vector<T> data_;
};

class Layer {
    public: 
        int numLayer, inFm, outFm, inW, coreW;
        std::string name;
        std::string type;
        vector_3d<int> inputData;
        vector_3d<int> outputData;
        vector_4d<int> weights;

        Layer (std::string,int,int,int,int,int);
        void print();
        void forward(); 
        void conv();
        void pool();
        void upsample();
        void printInputs();
        void printOutputs();   
};