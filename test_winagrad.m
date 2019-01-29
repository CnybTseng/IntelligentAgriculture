clear;clc;

% 8x3 => 8x4
G = [    1,       0,       0, 0;...
	-2/9.0,  -2/9.0,  -2/9.0, 0;...
	-2/9.0,   2/9.0,  -2/9.0, 0;...
	1/90.0,  1/45.0,  2/45.0, 0;...
	1/90.0, -1/45.0,  2/45.0, 0;...
	1/45.0,  1/90.0, 1/180.0, 0;...
	1/45.0, -1/90.0, 1/180.0, 0;...
	     0,       0,       1, 0];

% 8x8		 
B = [1,      0, -21/4.0,       0,  21/4.0,       0, -1, 0;
	 0,      1,       1, -17/4.0, -17/4.0,       1,  1, 0;
	 0,     -1,       1,  17/4.0, -17/4.0,      -1,  1, 0;
	 0,  1/2.0,   1/4.0,  -5/2.0,  -5/4.0,       2,  1, 0;
	 0, -1/2.0,   1/4.0,   5/2.0,  -5/4.0,      -2,  1, 0;
	 0,      2,       4,  -5/2.0,      -5,   1/2.0,  1, 0;
	 0,     -2,       4,   5/2.0,      -5,  -1/2.0,  1, 0;
	 0,     -1,       0,  21/4.0,       0, -21/4.0,  0, 1];

% 6x8 => 8x8	 
A = [1, 1,  1,  1,   1, 32,  32, 0;...
	 0, 1, -1,  2,  -2, 16, -16, 0;...
	 0, 1,  1,  4,   4,  8,   8, 0;...
	 0, 1, -1,  8,  -8,  4,  -4, 0;...
	 0, 1,  1, 16,  16,  2,   2, 0;...
	 0, 1, -1, 32, -32,  1,  -1, 1;...
	 0, 0,  0,  0,   0,  0,   0, 0;...
	 0, 0,  0,  0,   0,  0,   0, 0];

% 3x3 => 4x4	 
g = zeros(4,4);
g(1:3,1:3) = rand(3,3);

% 8x8
d = rand(8,8);

Y = A*((G*g*G').*(B*d*B'))*A';

G = G(:,1:3);
A = A(1:6,:);
g = g(1:3,1:3);
Y2 = A*((G*g*G').*(B*d*B'))*A';

output = rand(8,8,3);
inverse_output_per_channel(:,:,1) = A*output(:,:,1)*A';
inverse_output_per_channel(:,:,2) = A*output(:,:,2)*A';
inverse_output_per_channel(:,:,3) = A*output(:,:,3)*A';
inverse_output1 = sum(inverse_output_per_channel,3);

sum_output = sum(output,3);
inverse_output2 = A*sum_output*A';