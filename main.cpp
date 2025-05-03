//#include "Eigen/Dense"
#include <vector>
#include <limits>
#include <cstring>
#include <iostream>
using namespace Eigen;
namespace Optimization
{
    template<int nDims>
    class NelderMead {
    public:
        typedef Array<double, 1, nDims> FunctionParameters;
        typedef std::function<double(FunctionParameters)> ErrorFunction;
        NelderMead(ErrorFunction errorFunction, FunctionParameters initial,
                   double minError, double initialEdgeLength,
                   double shrinkCoeff = 1, double contractionCoeff = 0.5,
                   double reflectionCoeff = 1, double expansionCoeff = 1)
                : errorFunction(errorFunction), minError(minError),
                shrinkCoeff(shrinkCoeff), contractionCoeff(contractionCoeff),
                reflectionCoeff(reflectionCoeff), expansionCoeff(expansionCoeff),
                worstValueId(-1), secondWorstValueId(-1), bestValueId(-1)
        {
            this->errors = std::vector(nDims + 1, std::numeric_limits<double>::max());
            const double b = initialEdgeLength / (nDims * SQRT2) * (sqrt(nDims + 1) - 1);
            const double a = initialEdgeLength / SQRT2;
            this->values = initial.replicate(nDims + 1, 1);
            for (int i = 0; i < nDims; i++)
            {
                FunctionParameters simplexRow;
                simplexRow.setConstant(b);
                simplexRow(0, i) = a;
                simplexRow += initial;
                this->values.row(i+1) = simplexRow;
            }
        }
        void optimize()
        {
            for (int i = 0; i < nDims+1; i++)
            {
                this->errors.at(i) = this->errorFunction(this->values.row(i));
            }

            this->invalidateIdsCache();
            while (this->errors.at(this->bestValueId) > this->minError)
            {
                step();
                auto bestError = this->errorFunction(this->best());
                auto worstError = this->errorFunction(this->worst());
                std::cout << "Best error " << std::to_string(bestError) << " with : " << this->best();
                std::cout << " Worst error " << std::to_string(worstError) << " with : " << this->worst();
                std::cout << '\n';
            }
        }
        void step()
        {
            auto meanWithoutWorst = this->getMeanWithoutWorst();
            auto reflectionOfWorst = this->getReflectionOfWorst(meanWithoutWorst);
            auto reflectionError = this->errorFunction(reflectionOfWorst);
            FunctionParameters newValue = reflectionOfWorst;
            double newError = reflectionError;
            bool shrink = false;
            if (reflectionError < this->errors.at(this->bestValueId))
            {
                auto expansionValue = this->expansion(meanWithoutWorst, reflectionOfWorst);
                double expansionError = this->errorFunction(expansionValue);
                if (expansionError < this->errors.at(this->bestValueId))
                {
                    newValue = expansionValue;
                    newError = expansionError;
                }
            }
            else if (reflectionError > this->errors.at(this->worstValueId))
            {
                newValue = this->insideContraction(meanWithoutWorst);
                newError = this->errorFunction(newValue);

                if (newError > this->errors.at(this->worstValueId)) { shrink = true; }
            }
            else if (reflectionError > this->errors.at(this->secondWorstValueId))
            {
                newValue = this->outsideContraction(meanWithoutWorst);
                newError = this->errorFunction(newValue);
                if (newError > reflectionError) { shrink = true; }
            }
            else
            {
                newValue = reflectionOfWorst;
                newError = reflectionError;
            }
            if (shrink)
            {
                this->shrink();
                this->invalidateIdsCache();
                return;
            }
            this->values.row(this->worstValueId) = newValue;
            this->errors.at(this->worstValueId) = newError;
            this->invalidateIdsCache();
        }
        inline FunctionParameters worst() { return this->values.row(this->worstValueId); }
        inline FunctionParameters best() { return this->values.row(this->bestValueId); }
    private:
        void shrink()
        {
            auto bestVertex = this->values.row(this->bestValueId);
            for (int i = 0; i < nDims + 1; i++)
            {
                if (i == this->bestValueId) { continue; }
                this->values.row(i) = bestVertex + this->shrinkCoeff * (this->values.row(i) - bestVertex);
                this->errors.at(i) = this->errorFunction(this->values.row(i));
            }
        }
        inline FunctionParameters expansion(FunctionParameters meanWithoutWorst, FunctionParameters reflection)
        {
            return reflection + this->expansionCoeff * (reflection - meanWithoutWorst);
        }
        inline FunctionParameters insideContraction(FunctionParameters meanWithoutWorst)
        {
            return meanWithoutWorst - this->contractionCoeff * (meanWithoutWorst - this->worst());
        }
        inline FunctionParameters outsideContraction(FunctionParameters meanWithoutWorst)
        {
            return meanWithoutWorst + this->contractionCoeff * (meanWithoutWorst - this->worst());
        }
        FunctionParameters getReflectionOfWorst(FunctionParameters meanWithoutWorst)
        {
            return meanWithoutWorst + this->reflectionCoeff * (meanWithoutWorst - this->worst());
        }
        FunctionParameters getMeanWithoutWorst()
        {
            FunctionParameters mean(0);
            for (int i = 0; i < nDims + 1; i++)
            {
                if (i == this->worstValueId) { continue; }
                mean += this->values.row(i);
            }
            mean /= nDims;
            return mean;
        }
        void invalidateIdsCache() {
            double worstError = std::numeric_limits<double>::min();
            int worstId = -1;
            double secondWorstError = std::numeric_limits<double>::max();
            int secondWorstId = -1;
            double bestError = std::numeric_limits<double>::max();
            int bestId = -1;
            for (int i = 0; i < nDims + 1; i++)
            {
                auto error = this->errors.at(i);
                if (error > worstError)
                {
                    secondWorstError = worstError;
                    secondWorstId = worstId;
                    worstError = error;
                    worstId = i;
                }
                else if (error > secondWorstError)
                {
                    secondWorstError = error;
                    secondWorstId = i;
                }
                if (error < bestError)
                {
                    bestError = error;
                    bestId = i;
                }
            }
                       if (secondWorstId == -1)
            {
                secondWorstId = worstId;
            }
            this->bestValueId = bestId;
            this->worstValueId = worstId;
            this->secondWorstValueId = secondWorstId;
        }
        ErrorFunction errorFunction;
        Array<double, nDims + 1, nDims> values;
        std::vector<double> errors;
        int worstValueId;
        int secondWorstValueId;
        int bestValueId;
        double minError;
        double shrinkCoeff;
        double expansionCoeff;
        double contractionCoeff;
        double reflectionCoeff;
        const double SQRT2 = sqrt(2);
    };
}
