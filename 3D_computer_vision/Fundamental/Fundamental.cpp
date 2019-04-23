// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse
// Date:     2013/10/08

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    int Niter=100000; // Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;
    // --------------- TODO ------------
    // DO NOT FORGET NORMALIZATION OF POINTS

    int n = matches.size();
    //Indices of matches used to generate random k-samples of the set of matches
    vector<int> indices;
    for(size_t t = 0 ;t<n ; t++){
        indices.push_back(t);
    }

    //Construction of the normalization matrix
    float normvals[3][3] = {{0.001,0,0},{0,0.001,0},{0,0,1.0}};
    FMatrix<float,3,3> N(normvals);
    //Number of randomly selected matches at each step of Ransac
    const int k = 8;
    vector<int>training_set;
    int nmax = 0 ;//maximum number of inliers

    //Ransac Loop
    for(size_t t = 0 ;t<Niter ; t++){
        //generate a k-sample of matches, here k = 8
        random_shuffle(indices.begin(),indices.end());
        training_set = indices;
        training_set.resize(k);// contains the indices of a randomly generated k-sample of matches

        //Construction of the matrix A representing the linear system A *f = 0
        FMatrix<float,3,3> F;
        FMatrix<float,k+1,9> A;
        float x1,y1,x2,y2;
        vector<int>::const_iterator it;
        int i ;//index of current match
        int line = 0;//index of line being filled in  A
        for(it = training_set.begin(); it!=training_set.end(); it++){
            i = *it;
            FVector<float,3> p1(matches[i].x1,matches[i].y1,1.0);
            FVector<float,3> p2(matches[i].x2,matches[i].y2,1.0);
            FVector<float,3> X1 = N*p1;
            FVector<float,3> X2 = N*p2;

            x1 = X1.x();
            y1 = X1.y();
            x2 = X2.x();
            y2 = X2.y();

            A(line,0) = x1*x2;
            A(line,1) = x1*y2;
            A(line,2) = x1;
            A(line,3) = y1*x2;
            A(line,4) = y1*y2;
            A(line,5) = y1;
            A(line,6) = x2;
            A(line,7) = y2;
            A(line,8) = 1.0;
            line++;
        };
        for(int j = 0;j < 9 ; j++){
            A(k,j)= 0;
        }
        //Singular value decomposition
        FVector<float,9> S;
        FMatrix<float,9,9> U,Vt;
        svd(A,U,S,Vt,true);
        //The solution of the system is the eigenvector of A.T*A associated with its smallest eigenvalue
        FVector<float,9> fvect = Vt.getRow(k);
        FMatrix<float,3,3> Fn;

        //Fill normalized F with the solutions

        Fn(0,0) = fvect[0];
        Fn(1,0) = fvect[1];
        Fn(2,0) = fvect[2];
        Fn(0,1) = fvect[3];
        Fn(1,1) = fvect[4];
        Fn(2,1) = fvect[5];
        Fn(0,2) = fvect[6];
        Fn(1,2) = fvect[7];
        Fn(2,2) = fvect[8];

//      Impose the rank 2 constraint on Fn by setting it's last singular value to 0
        FVector<float,3> S1;
        FMatrix<float,3,3> U1,Vt1,Sigma;
        svd(Fn,U1,S1,Vt1,true);
        S1.z() = 0;
        Sigma = Diagonal(S1);
        cout << "Sigma :" << Sigma <<endl;
        Fn = U1*Sigma*Vt1;
        cout << "Fn :" << Fn << endl;
        //Denormalize F
        F = N*Fn*N;

        //Search for the inliers for the current model in the set of matches.
        //Compare distance of x2 from epipolar line associated with x1 (F.T* x1) with the treshold distMax
        vector<Match>::const_iterator iter;
        int n_inliers = 0;//numbers of inliers for the current model. The 8 matches on which the model is computed are considered inliers
        vector<int> inliers;
        int match_index = 0;//used to store the index of inliers.
        for(iter = matches.begin();  iter!= matches.end(); iter++){

                FVector<float,3> X1(iter->x1,iter->y1,1.0);
                FVector<float,3> X2(iter->x2,iter->y2,1.0);
                FVector<float,3> U = F*X1;

                float norm = sqrt(U.x()*U.x() + U.y()*U.y()); //  normalize vector to compute the distance to the epipolar line
                float dist = abs(X2*U)/norm; //Distance of (x2,y2) to the line perpendicular to u
                if(dist < distMax){
                    n_inliers ++;
                    inliers.push_back(match_index);
                };
                match_index ++;
        }
        double tmp;
        //Updates the set of inliers with the greatest cardinal
        if(n_inliers > nmax){
            bestF = F;
            bestInliers = inliers;
            nmax = n_inliers;
            //Adjusts Niter dynamically
            tmp = (ceil(log(BETA)/log(1 - pow((float(n_inliers)/float(n)),8))));
            if(abs(tmp) < double(Niter)){
                Niter = tmp;
            }
        }   
    }


    //Refining F with best inliers using least square minimization associated with the linear system Af=0
    //As above the solution of the least square problem is the eigenvalue of A.T * A associated with the smallest eigenvalue
    int ninliers = bestInliers.size();
    Matrix<float> A(ninliers,9);
    float x1,y1,x2,y2;
    vector<int>::const_iterator it;
    int i ;//index of current match
    int line = 0;//index of line being filled in  A
    for(it = bestInliers.begin(); it!=bestInliers.end(); it++){
        i = *it;
        FVector<float,3> p1(matches[i].x1,matches[i].y1,1.0);
        FVector<float,3> p2(matches[i].x2,matches[i].y2,1.0);
        FVector<float,3> X1 = N*p1;
        FVector<float,3> X2 = N*p2;

        x1 = X1.x();
        y1 = X1.y();
        x2 = X2.x();
        y2 = X2.y();

        A(line,0) = x1*x2;
        A(line,1) = x1*y2;
        A(line,2) = x1;
        A(line,3) = y1*x2;
        A(line,4) = y1*y2;
        A(line,5) = y1;
        A(line,6) = x2;
        A(line,7) = y2;
        A(line,8) = 1.0;
        line++;
    };


    //Singular value decomposition
    Vector<float> S(ninliers);
    Matrix<float> U(ninliers,ninliers);
    Matrix<float> Vt(9,9);
    cout << "A :" << A << endl;
    svd(A,U,S,Vt,true);

    //The solution of the system is the eigenvector of A.T*A associated with its smallest eigenvalue
    Vector<float> fvect(9);
    fvect = Vt.getRow(8);

    //Fill normalized F with the solutions
    FMatrix<float,3,3> Fn;

    Fn(0,0) = fvect[0];
    Fn(1,0) = fvect[1];
    Fn(2,0) = fvect[2];
    Fn(0,1) = fvect[3];
    Fn(1,1) = fvect[4];
    Fn(2,1) = fvect[5];
    Fn(0,2) = fvect[6];
    Fn(1,2) = fvect[7];
    Fn(2,2) = fvect[8];

    //Impose the rank 2 constraint on Fn
    FVector<float,3> S1;
    FMatrix<float,3,3> U1,Vt1,Sigma;
    svd(Fn,U1,S1,Vt1,true);
    S1.z() = 0;
    Sigma = Diagonal(S1);
    Fn = U1*Sigma*Vt1;
    //Denormalize F
    bestF = N*Fn*N;


    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);
    return bestF;
}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {


    while(true) {
        int x,y;
        int w1 = I1.width();
        int w2 = I2.width();
        cout << "Click somewhere on one of the images" << endl;

        if(getMouse(x,y) == 3)
            break;

        // --------------- TODO ------------

        //Display the clicked point
        fillCircle(x,y,15,RED);

        //Display epipolar line on the other image;
        FVector<float,3> u;
        //If left image is clicked
        if(x<w1){
            FVector<float,3> p(x,y,1);
            u = F * p;
            int x1 = w1;
            int y1 = -(1/u.y())*(u.x()*0 + u.z());
            int x2 = w1+w2-1;
            int y2 = -(1/u.y())*(u.x()*(w2) + u.z());
            drawLine(x1,y1,x2,y2,GREEN);
        }
        //If right image is clicked
        if(w1<x){
            FVector<float,3> p(x-w1,y,1);
            u = transpose(F) * p;
            int x1 = 0;
            int y1 = -(1.0/u.y())*(u.x()*0 + u.z());
            int x2 = w1-1;
            int y2 = -(1.0/u.y())*(u.x()*(w1) + u.z());
            drawLine(x1,y1,x2,y2,GREEN);
        }
    }
}

int main(int argc, char* argv[])
{


    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    cout << " matches: " << matches.size() << endl;
    click();
    
    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);        
    }
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
