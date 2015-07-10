// -*- C++ -*-

// A basic LSTM implementation in C++. All you should need is clstm.cpp and
// clstm.h. Library dependencies are limited to a small subset of STL and
// Eigen/Dense

#ifndef ocropus_lstm_
#define ocropus_lstm_

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>
#include <memory>
#include <map>
#include <Eigen/Dense>
#include <random>

namespace ocropus {
using std::string;
using std::vector;
using std::map;
using std::shared_ptr;
using std::unique_ptr;
using std::function;

#ifdef LSTM_DOUBLE
typedef double Float;
typedef Eigen::VectorXi iVec;
typedef Eigen::VectorXd Vec;
typedef Eigen::MatrixXd Mat;
#else
typedef float Float;
typedef Eigen::VectorXi iVec;
typedef Eigen::VectorXf Vec;
typedef Eigen::MatrixXf Mat;
#endif

typedef vector<Mat> Sequence;
typedef vector<int> Classes;
typedef vector<Classes> BatchClasses;

// These macros define the major matrix operations used
// in CLSTM. They are here for eventually converting the
// inner loops of CLSTM from Eigen::Matrix to Eigen::Tensor
// (which uses different and incompatible notation)
#define DOT(M, V) ((M) *(V))
#define MATMUL(A, B) ((A) *(B))
#define MATMUL_TR(A, B) ((A).transpose() * (B))
#define MATMUL_RT(A, B) ((A) *(B).transpose())
#define EMUL(U, V) ((U).array() * (V).array()).matrix()
#define EMULV(U, V) ((U).array() * (V).array()).matrix()
#define TRANPOSE(U) ((U).transpose())
#define ROWS(A) (A).rows()
#define COLS(A) (A).cols()
#define COL(A, b) (A).col(b)
#define MAPFUN(M, F) ((M).unaryExpr(ptr_fun(F)))
#define MAPFUNC(M, F) ((M).unaryExpr(F))
#define SUMREDUCE(M) float(M.sum())
#define BLOCK(A, i, j, n, m) (A).block(i, j, n, m)

inline void ADDCOLS(Mat &m, Vec &v) {
    for (int i = 0; i < COLS(m); i++) {
        for (int j = 0; j < ROWS(m); j++) {
            m(j, i) += v(j);
        }
    }
}
inline void randgauss(Mat &m) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> randn;
    for (int i = 0; i < ROWS(m); i++)
        for (int j = 0; j < COLS(m); j++)
            m(i, j) = randn(gen);
}
inline void randgauss(Vec &v) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> randn;
    for (int i = 0; i < ROWS(v); i++)
        v(i) = randn(gen);
}
inline void randinit(Mat &m, float s) {
    m.setRandom();
    m = (2*s*m).array()-s;
}
inline void randinit(Vec &m, float s) {
    m.setRandom();
    m = (2*s*m).array()-s;
}
inline void randinit(Mat &m, int no, int ni, float s) {
    m.resize(no, ni);
    randinit(m, s);
}
inline void randinit(Vec &m, int no, float s) {
    m.resize(no);
    randinit(m, s);
}
inline void zeroinit(Mat &m, int no, int ni) {
    m.resize(no, ni);
    m.setZero();
}
inline void zeroinit(Vec &m, int no) {
    m.resize(no);
    m.setZero();
}



inline void resize(Sequence &seq, int nsteps, int dims, int bs) {
    seq.resize(nsteps);
    for (int i=0; i<nsteps; i++) seq[i].resize(dims,bs);
}

inline int size(Sequence &seq, int dim) {
    if (dim==0) return seq.size();
    if (dim==1) return seq[0].rows();
    if (dim==2) return seq[0].cols();
    assert(0 && "bad dim ins size");
}

inline Vec timeslice(const Sequence &s, int i, int b=0) {
    Vec result(s.size());
    for (int t = 0; t < s.size(); t++)
        result[t] = s[t](i, b);
    return result;
}

struct VecMat {
    Vec *vec = 0;
    Mat *mat = 0;
    VecMat() {
    }
    VecMat(Vec *vec) {
        this->vec = vec;
    }
    VecMat(Mat *mat) {
        this->mat = mat;
    }
};

struct ITrainable {
    virtual ~ITrainable() {
    }
    string name = "";
    virtual const char *kind() = 0;

    // Learning rate and momentum used for training.
    Float learning_rate = 1e-4;
    Float momentum = 0.9;
    enum Normalization : int {
        NORM_NONE, NORM_LEN, NORM_BATCH, NORM_DFLT = NORM_NONE,
    } normalization = NORM_DFLT;

    // The attributes array contains parameters for constructing the
    // network, as well as information necessary for loading and saving
    // networks.
    map<string, string> attributes;
    string attr(string key, string dflt="") {
        auto it = attributes.find(key);
        if (it == attributes.end()) return dflt;
        return it->second;
    }
    int iattr(string key, int dflt=-1) {
        auto it = attributes.find(key);
        if (it == attributes.end()) return dflt;
        return std::stoi(it->second);
    }
    double dattr(string key, double dflt=0.0) {
        auto it = attributes.find(key);
        if (it == attributes.end()) return dflt;
        return std::stof(it->second);
    }
    int irequire(string key) {
        auto it = attributes.find(key);
        if (it == attributes.end()) {
            assert(0 && "no key in params");
        }
        return std::stoi(it->second);
    }
    void set(string key, string value) {
        attributes[key] = value;
    }
    void set(string key, int value) {
        attributes[key] = std::to_string(value);
    }
    void set(string key, double value) {
        attributes[key] = std::to_string(value);
    }

    // Learning rates
    virtual void setLearningRate(Float lr, Float momentum) = 0;

    // Main methods for forward and backward propagation
    // of activations.
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void update() = 0;

    virtual int idepth() {
        return -9999;
    }
    virtual int odepth() {
        return -9999;
    }

    virtual void initialize() {
        // this gets initialization parameters
        // out of the attributes array
    }

    // These are convenience functions for initialization
    virtual void init(int no, int ni) final {
        set("ninput", ni);
        set("noutput", no);
        initialize();
    }
    virtual void init(int no, int nh, int ni) final {
        set("ninput", ni);
        set("nhidden", nh);
        set("noutput", no);
        initialize();
    }
    virtual void init(int no, int nh2, int nh, int ni) final {
        set("ninput", ni);
        set("nhidden", nh);
        set("nhidden2", nh2);
        set("noutput", no);
        initialize();
    }
};

struct INetwork;
typedef shared_ptr<INetwork> Network;

struct INetwork : virtual ITrainable {
    // Networks have input and output "ports" for sequences
    // and derivatives. These are propagated in forward()
    // and backward() methods.
    Sequence inputs, d_inputs;
    Sequence outputs, d_outputs;

    // Some networks have subnetworks. They should be
    // stored in the `sub` vector. That way, functions
    // like `save` can automatically traverse the tree
    // of networks. Together with the `name` field,
    // this forms a hierarchical namespace of networks.
    vector<Network > sub;

    // Parameters specific to softmax.
    Float softmax_floor = 1e-5;
    bool softmax_accel = false;

    virtual ~INetwork() {
    }

    std::function<void(INetwork*)> initializer = [] (INetwork*){};
    virtual void initialize() {
        // this gets initialization parameters
        // out of the attributes array
        initializer(this);
    }

    // Expected number of input/output features.
    virtual int ninput() {
        return -999999;
    }
    virtual int noutput() {
        return -999999;
    }

    // Add a network as a subnetwork.
    virtual void add(Network net) {
        sub.push_back(net);
    }

    // Hooks to iterate over the weights and states of this network.
    typedef function<void (const string &, VecMat, VecMat)> WeightFun;
    typedef function<void (const string &, Sequence *)> StateFun;
    virtual void myweights(const string &prefix, WeightFun f) {
    }
    virtual void mystates(const string &prefix, StateFun f) {
    }

    // Hooks executed prior to saving and after loading.
    // Loading iterates over the weights with the `weights`
    // methods and restores only the weights. `postLoad`
    // allows classes to update other internal state that
    // depends on matrix size.
    virtual void preSave() {
    }
    virtual void postLoad() {
    }

    // Set the learning rate for this network and all subnetworks.
    virtual void setLearningRate(Float lr, Float momentum) {
        this->learning_rate = lr;
        this->momentum = momentum;
        for (int i = 0; i < sub.size(); i++)
            sub[i]->setLearningRate(lr, momentum);
    }

    void info(string prefix);
    void weights(const string &prefix, WeightFun f);
    void states(const string &prefix, StateFun f);
    void networks(const string &prefix, function<void (string, INetwork*)>);
    Sequence *getState(string name);
    // special method for LSTM and similar networks, returning the
    // primary internal state sequence
    Sequence *getState() {
        assert(0 && "unimplemented");
    };
    void save(const char *fname);
    void load(const char *fname);
};

// standard layer types
INetwork *make_SigmoidLayer();
INetwork *make_SoftmaxLayer();
INetwork *make_ReluLayer();
INetwork *make_Stacked();
INetwork *make_Reversed();
INetwork *make_Parallel();
INetwork *make_LSTM();
INetwork *make_NPLSTM();
INetwork *make_BidiLayer();

// setting inputs and outputs
void set_inputs(INetwork *net, Sequence &inputs);
void set_targets(INetwork *net, Sequence &targets);
void set_targets_accelerated(INetwork *net, Sequence &targets);

// single sequence training functions
void train(INetwork *net, Sequence &xs, Sequence &targets);

// instantiating layers and networks

typedef std::function<INetwork*(void)> ILayerFactory;
extern map<string, ILayerFactory> layer_factories;
Network make_layer(const string &kind);

struct String : public std::string {
    String() {
    }
    String(const char *s) : std::string(s) {
    }
    String(const std::string &s) : std::string(s) {
    }
    String(int x) : std::string(std::to_string(x)) {
    }
    String(double x) : std::string(std::to_string(x)) {
    }
    double operator+() { return atof(this->c_str()); }
    operator int() {
        return atoi(this->c_str());
    }
    operator double() {
        return atof(this->c_str());
    }
};
struct Assoc : std::map<std::string, String> {
    using std::map<std::string, String>::map;
    Assoc(const string &s);
    String at(const std::string &key) const {
        auto it = this->find(key);
        if (it == this->end()) assert(0 && "key not found");
        return it->second;
    }
};
typedef std::vector<Network> Networks;
Network layer(
    const string &kind,
    int ninput, int noutput,
    const Assoc &args,
    const Networks &subs
    );

typedef std::function<Network(const Assoc &)> INetworkFactory;
extern map<string, INetworkFactory> network_factories;
Network make_net(const string &kind, const Assoc &params);
Network make_net_init(const string &kind, const std::string &params);

}

#endif
