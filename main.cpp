#include <vector>
#include <limits>
#include <iostream>
#include <functional>
#include <cmath>
#include <numeric>

using namespace std;

template<int nDims>
class NelderMead {
public:
    typedef vector<double> FunctionParameters;
    typedef function<double(const FunctionParameters&)> ErrorFunction;

    NelderMead(ErrorFunction errorFunction, 
               const FunctionParameters& initial,
               double minError, 
               double initialEdgeLength,
               double shrinkCoeff = 0.5,
               double contractionCoeff = 0.5,
               double reflectionCoeff = 1.0,
               double expansionCoeff = 2.0) :
        errorFunction(errorFunction),
        minError(minError),
        shrinkCoeff(shrinkCoeff),
        contractionCoeff(contractionCoeff),
        reflectionCoeff(reflectionCoeff),
        expansionCoeff(expansionCoeff)
    {
        this->errors = vector<double>(nDims + 1, numeric_limits<double>::max());
        this->values = vector<FunctionParameters>(nDims + 1, FunctionParameters(nDims));
        
        const double b = initialEdgeLength / (nDims * sqrt(2)) * (sqrt(nDims + 1) - 1);
        const double a = initialEdgeLength / sqrt(2);
        
        for (int i = 0; i < nDims + 1; i++) {
            if (i == 0) {
                this->values[i] = initial;
            } else {
                FunctionParameters simplexRow(nDims, b);
                simplexRow[i-1] = a;
                for (int j = 0; j < nDims; j++) {
                    this->values[i][j] = simplexRow[j] + initial[j];
                }
            }
        }
        
        for (int i = 0; i < nDims + 1; i++) {
            this->errors[i] = this->errorFunction(this->values[i]);
        }
        
        invalidateIdsCache();
    }

    void optimize(int maxIterations = 1000) {
        int iteration = 0;
        while (this->errors[this->bestValueId] > this->minError && iteration < maxIterations) {
            step();
            iteration++;
            
            if (iteration % 10 == 0) {
                cout << "Iteration " << iteration 
                     << ": Best error = " << this->errors[this->bestValueId]
                     << " at [";
                for (double val : this->best()) {
                    cout << val << " ";
                }
                cout << "]" << endl;
            }
        }
    }

    const FunctionParameters& best() const { return this->values[this->bestValueId]; }
    const FunctionParameters& worst() const { return this->values[this->worstValueId]; }
    double bestError() const { return this->errors[bestValueId]; }

private:
    void step() {
        FunctionParameters meanWithoutWorst = getMeanWithoutWorst();
        FunctionParameters reflection = getReflectionOfWorst(meanWithoutWorst);
        double reflectionError = errorFunction(reflection);
        
        FunctionParameters newValue;
        double newError;
        bool shrink = false;

        if (reflectionError < this->errors[bestValueId]) {
            FunctionParameters expansionValue = expansion(meanWithoutWorst, reflection);
            double expansionError = errorFunction(expansionValue);
            
            if (expansionError < reflectionError) {
                newValue = expansionValue;
                newError = expansionError;
            } else {
                newValue = reflection;
                newError = reflectionError;
            }
        }
        else if (reflectionError > this->errors[secondWorstValueId]) {
            if (reflectionError <= this->errors[worstValueId]) {
                newValue = outsideContraction(meanWithoutWorst);
                newError = errorFunction(newValue);
                
                if (newError > reflectionError) {
                    shrink = true;
                }
            } else {
                newValue = insideContraction(meanWithoutWorst);
                newError = errorFunction(newValue);
                
                if (newError > this->errors[worstValueId]) {
                    shrink = true;
                }
            }
        }
        else {
            newValue = reflection;
            newError = reflectionError;
        }

        if (shrink) {
            this->shrink();
        } else {
            this->values[worstValueId] = newValue;
            this->errors[worstValueId] = newError;
        }
        
        invalidateIdsCache();
    }

    void shrink() {
        const FunctionParameters& bestVertex = this->values[bestValueId];
        for (int i = 0; i < nDims + 1; i++) {
            if (i == bestValueId) continue;
            
            for (int j = 0; j < nDims; j++) {
                this->values[i][j] = bestVertex[j] + 
                    shrinkCoeff * (this->values[i][j] - bestVertex[j]);
            }
            this->errors[i] = errorFunction(this->values[i]);
        }
    }

    FunctionParameters expansion(const FunctionParameters& mean, 
                               const FunctionParameters& reflection) const {
        FunctionParameters result(nDims);
        for (int i = 0; i < nDims; i++) {
            result[i] = mean[i] + expansionCoeff * (reflection[i] - mean[i]);
        }
        return result;
    }

    FunctionParameters insideContraction(const FunctionParameters& mean) const {
        FunctionParameters result(nDims);
        const FunctionParameters& w = worst();
        for (int i = 0; i < nDims; i++) {
            result[i] = mean[i] - contractionCoeff * (mean[i] - w[i]);
        }
        return result;
    }

    FunctionParameters outsideContraction(const FunctionParameters& mean) const {
        FunctionParameters result(nDims);
        const FunctionParameters& w = worst();
        for (int i = 0; i < nDims; i++) {
            result[i] = mean[i] + contractionCoeff * (mean[i] - w[i]);
        }
        return result;
    }

    FunctionParameters getReflectionOfWorst(const FunctionParameters& mean) const {
        FunctionParameters result(nDims);
        const FunctionParameters& w = worst();
        for (int i = 0; i < nDims; i++) {
            result[i] = mean[i] + reflectionCoeff * (mean[i] - w[i]);
        }
        return result;
    }

    FunctionParameters getMeanWithoutWorst() const {
        FunctionParameters mean(nDims, 0.0);
        for (int i = 0; i < nDims + 1; i++) {
            if (i == worstValueId) continue;
            for (int j = 0; j < nDims; j++) {
                mean[j] += this->values[i][j];
            }
        }
        for (int j = 0; j < nDims; j++) {
            mean[j] /= nDims;
        }
        return mean;
    }

    void invalidateIdsCache() {
        worstValueId = 0;
        bestValueId = 0;
        secondWorstValueId = 0;

        for (int i = 1; i < nDims + 1; i++) {
            if (errors[i] > errors[worstValueId]) {
                secondWorstValueId = worstValueId;
                worstValueId = i;
            } else if (errors[i] > errors[secondWorstValueId]) {
                secondWorstValueId = i;
            }

            if (errors[i] < errors[bestValueId]) {
                bestValueId = i;
            }
        }

        if (secondWorstValueId == worstValueId) {
            for (int i = 0; i < nDims + 1; i++) {
                if (i != worstValueId && i != bestValueId) {
                    secondWorstValueId = i;
                    break;
                }
            }
        }
    }

    ErrorFunction errorFunction;
    double minError;
    double shrinkCoeff;
    double contractionCoeff;
    double reflectionCoeff;
    double expansionCoeff;

    vector<FunctionParameters> values;
    vector<double> errors;
    int worstValueId;
    int secondWorstValueId;
    int bestValueId;
};

int main() {
    auto quadratic = [](const vector<double>& x) {
        return x[0]*x[0] + x[1]*x[1] + 2*x[0] + 4*x[1] + 5;
    };

    vector<double> initial = {1.0, 1.0};

    NelderMead<2> optimizer(quadratic, initial, 1e-6, 1.0);

    optimizer.optimize();

    cout << "\nOptimization complete!" << endl;
    cout << "Minimum found at: [";
    for (double val : optimizer.best()) {
        cout << val << " ";
    }
    cout << "]" << endl;
    cout << "Function value: " << optimizer.bestError() << endl;

    return 0;
}