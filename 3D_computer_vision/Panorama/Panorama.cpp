// Imagine++ project
// Project:  Panorama
// Author:   Pascal Monasse
// Date:     2013/10/08

#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <sstream>
using namespace Imagine;
using namespace std;

// Record clicks in two images, until right button click
void getClicks(Window w1, Window w2,
               vector<IntPoint2>& pts1, vector<IntPoint2>& pts2) {
    // ------------- TODO/A completer ----------
    IntPoint2 p1,p2;
    int count =0;

    int sw1,sw2;
    int key1,key2,e; //No right click on mac
    int button1,button2 = 0;
    cout << "To create panorama, select at least four couple of matching points by clicking them one after the other" << endl;
    cout << "Press e to begin" <<endl;
    e = anyGetKey(w2,sw2);

    while(count < 4){
        cout << "Click matching points on the images" <<  endl;
        cout << "Need at least " << 4-count << " more matching points" << endl;

        button2 = anyGetMouse(p2,w2,sw2);
        pts2.push_back(p2);//pts2 are the clicked points on window w2

        button1 = anyGetMouse(p1,w1,sw1);
        pts1.push_back(p1);//pts1 are the clicked points on window w2


        count++;
        cout << "Selected points : " <<endl;
        cout << "p1 : " << p1 << endl;
        cout << "p2 : " << p2 << endl;

        cout << count << " points where clicked until now" <<endl;

    }
    while((button1 != 3) && (button2!=3) && (key1 != 101) && (key2 != 101)){//If the computer has no right click the user can press e
        cout << "Click matching points on the images" <<  endl;
        cout << "Use right click or press e to end" <<endl;


        if((button2 = anyGetMouse(p2,w2,sw2)) ==3){
                break;
        };

        pts2.push_back(p2);

        if((button1 = anyGetMouse(p1,w1,sw1)) == 3){
                break;
        };

        pts1.push_back(p1);

        if((key1 = anyGetKey(w1,sw1)) ==  e || (key2 = anyGetKey(w2,sw2)) == e){
            break;
        }
        cout << key1 << endl;
        cout << key2 << endl;
        count ++;
        cout << "Selected points : " <<endl;
        cout << "p1 : " << p1 << endl;
        cout << "p2 : " << p2 << endl;

        cout << count << " points where clicked until now" <<endl;

    }
}

// Return homography compatible with point matches
Matrix<float> getHomography(const vector<IntPoint2>& pts1,
                            const vector<IntPoint2>& pts2) {
    size_t n = min(pts1.size(), pts2.size());
    if(n<4) {
        cout << "Not enough correspondences: " << n << endl;
        return Matrix<float>::Identity(3);
    }
    Matrix<double> A(2*n,8);
    Vector<double> b(2*n);
    // ------------- TODO/A completer ----------
    for(int i = 0; i < n; i++){
        IntPoint2 p1 = pts1[i];
        IntPoint2 p2 = pts2[i];
        int xi= pts1[i][0];
        int yi= pts1[i][1];
        int xpi= pts2[i][0];
        int ypi= pts2[i][1];
        A(2*i,0) = xi;
        A(2*i,1) = yi;
        A(2*i,2) =1;
        A(2*i,3) =0;
        A(2*i,4) =0;
        A(2*i,5) =0;
        A(2*i,6) =-xpi*xi;
        A(2*i,7) =-xpi*yi;

        A(2*i+1,0) =0;
        A(2*i+1,1) =0;
        A(2*i+1,2) =0;
        A(2*i+1,3) =xi;
        A(2*i+1,4) =yi;
        A(2*i+1,5) =1;
        A(2*i+1,6) =-ypi*xi;
        A(2*i+1,7) =-ypi*yi;

        b[2*i] =xpi;
        b[2*i+1] =ypi;
    }
    cout << "A" << A << endl;


    Vector<double> B(8);
    B = linSolve(A, b);
    Matrix<float> H(3, 3);
    H(0,0)=B[0]; H(0,1)=B[1]; H(0,2)=B[2];
    H(1,0)=B[3]; H(1,1)=B[4]; H(1,2)=B[5];
    H(2,0)=B[6]; H(2,1)=B[7]; H(2,2)=1;

    // Sanity check
    for(size_t i=0; i<n; i++) {
        float v1[]={(float)pts1[i].x(), (float)pts1[i].y(), 1.0f};
        float v2[]={(float)pts2[i].x(), (float)pts2[i].y(), 1.0f};
        Vector<float> x1(v1,3);
        Vector<float> x2(v2,3);
        x1 = H*x1;
        cout << x1[1]*x2[2]-x1[2]*x2[1] << ' '
             << x1[2]*x2[0]-x1[0]*x2[2] << ' '
             << x1[0]*x2[1]-x1[1]*x2[0] << endl;
    }
    return H;
}

// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0, float& y0, float& x1, float& y1, float x, float y) {
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;    
}

// Panorama construction
void panorama(const Image<Color,2>& I1, const Image<Color,2>& I2,
              Matrix<float> H) {
    Vector<float> v(3);
    float x0=0, y0=0, x1=I2.width(), y1=I2.height();
    float x_ur, y_ur, x_lr,y_lr; //upper right and lower right corners of transformed I1

    v[0]=0; v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    x_ur = v[0];
    y_ur = v[1];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    x_lr = v[0];
    y_lr = v[1];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=0; v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    cout << "x0 x1 y0 y1=" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1<<endl;
    Image<Color> I(int(x1-x0), int(y1-y0));
    y_ur = y_ur - y0;
    setActiveWindow( openWindow(I.width(), I.height()) );

    I.fill(WHITE);
    // ------------- TODO/A completer ----------


//Draw I2
    for(int x = 0; x < I2.width(); x ++){
        for(int y = 0 ; y < I2.height();y++){
            int x_new  = x+I.width()-I2.width();
            int y_new  = y+y_ur;
            I(x_new,y_new) = I2(x,y);
        }
    }

//Draw I1
    Matrix<float> Hp = inverse(H); // We prefer to use the inverse to avoid sampling issues

    for(int x = 0; x < I.width(); x++){
        for(int y = 0 ; y < I.height(); y ++){
            v[0] = x+x0; v[1] = y+y0; v[2] = 1;
            v = Hp*v;
            v /= v[2];
            int x_new = int(v[0]);
            int y_new = int(v[1]);

            if((0< x_new) && (x_new <I1.width()) && (0< y_new) && (y_new < I1.height())){
                if(I(x,y) != WHITE){

                    I(x,y).r() = (I(x,y).r()+I1(x_new,y_new).r())/2.;
                    I(x,y).g() = (I(x,y).g()+I1(x_new,y_new).g())/2.;
                    I(x,y).b() = (I(x,y).b()+I1(x_new,y_new).b())/2.;

                }
                else{
                    I(x,y) = I1(x_new,y_new);
                }
            }
        }
    }

    int x_max = 0;int y_max = 0;

    display(I,0,0);
}

// Main function
int main(int argc, char* argv[]) {
    const char* s1 = argc>1? argv[1]: srcPath("image0006.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("image0007.jpg");

    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load the images" << endl;
        return 1;
    }
    Window w1 = openWindow(I1.width(), I1.height(), s1);
    display(I1,0,0);
    Window w2 = openWindow(I2.width(), I2.height(), s2);
    setActiveWindow(w2);
    display(I2,0,0);    

    // Get user's clicks in images
    vector<IntPoint2> pts1, pts2;
    getClicks(w1, w2, pts1, pts2);


    vector<IntPoint2>::const_iterator it;
    cout << "pts1="<<endl;
    for(it=pts1.begin(); it != pts1.end(); it++)
        cout << *it << endl;
    cout << "pts2="<<endl;
    for(it=pts2.begin(); it != pts2.end(); it++)
        cout << *it << endl;

    // Compute homography
    Matrix<float> H = getHomography(pts1, pts2);
    cout << "H=" << H/H(2,2);

    // Apply homography
    panorama(I1, I2, H);

    endGraphics();
    return 0;
}
