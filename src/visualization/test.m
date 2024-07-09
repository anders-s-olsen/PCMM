theta = [0,pi/2,pi/2,pi,pi];
theta = theta+rand(1,5)*0.5;

P = exp(1i*(theta-theta'));
[U11,L1]=eig(P); % only 1 non-zero eigenvalue
[U2,L2]=eig(real(P)); % 2 non-zero eigenvalues
[U3,L3]=eig(imag(P)); % 2 non-zero eigenvalues
