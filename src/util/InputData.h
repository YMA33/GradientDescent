#ifndef INPUTDATA_H_
#define INPUTDATA_H_

class InputData{
public:
    int num_tuples;
    int gradient_size;
    int num_classes;
    char* filename;
    
    InputData(int num_tuples, int gradient_size, int num_classes, char* filename);

};

InputData::InputData(int n_tuples, int grad_size, int n_classes, char* fname){
    num_tuples = n_tuples;
    gradient_size = grad_size;
    num_classes = n_classes;
    filename = fname;
}

#endif /* INPUTDATA_H_ */
