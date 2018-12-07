% transform filter matrix.
syms g00 g01 g02 g10 g11 g12 g20 g21 g22;

g0 = [g00 g01 g02 g10];
g1 = [g10 g11 g12 g20];
g2 = [g20 g21 g22 g12];

[w0 w1 w2 w3 w4 w5 w6 w7] = kernel_transform(g0,g1,g2);

w0123 = [w0;w1;w2;w3];
w0123 = transpose(w0123);
w0 = w0123(1,:);
w1 = w0123(2,:);
w2 = w0123(3,:);
w3 = w0123(4,:);

w4567 = [w4;w5;w6;w7];
w4567 = transpose(w4567);
w4 = w4567(1,:);
w5 = w4567(2,:);
w6 = w4567(3,:);
w7 = w4567(4,:);

[wg00 wg10 wg20 wg30 wg40 wg50 wg60 wg70] = kernel_transform(w0,w1,w2);
[wg01 wg11 wg21 wg31 wg41 wg51 wg61 wg71] = kernel_transform(w4,w5,w6);

for i=1:4
	transform_g(i,1)=wg00(i);
	transform_g(i,2)=wg10(i);
	transform_g(i,3)=wg20(i);
	transform_g(i,4)=wg30(i);
	transform_g(i,5)=wg40(i);
	transform_g(i,6)=wg50(i);
	transform_g(i,7)=wg60(i);
	transform_g(i,8)=wg70(i);
end

for i=5:8
	transform_g(i,1)=wg01(i-4);
	transform_g(i,2)=wg11(i-4);
	transform_g(i,3)=wg21(i-4);
	transform_g(i,4)=wg31(i-4);
	transform_g(i,5)=wg41(i-4);
	transform_g(i,6)=wg51(i-4);
	transform_g(i,7)=wg61(i-4);
	transform_g(i,8)=wg71(i-4);
end

fp = fopen('weight_transform-2.txt','w');
for y = 1:8
	for x = 1:8
		fprintf(fp,'%s\n',transform_g(y,x));
	end
	fprintf(fp,'\n\n\n');
end
fclose(fp);

% transform data matrix.
syms d00 d01 d02 d03 d04 d05 d06 d07 d10 d11 d12 d13 d14 d15 d16 d17 d20 d21 d22 d23 d24 d25 d26 d27 d30 d31 d32 d33 d34 d35 d36 d37 d40 d41 d42 d43 d44 d45 d46 d47 d50 d51 d52 d53 d54 d55 d56 d57 d60 d61 d62 d63 d64 d65 d66 d67 d70 d71 d72 d73 d74 d75 d76 d77;
dd = [d00 d01 d02 d03 d04 d05 d06 d07; d10 d11 d12 d13 d14 d15 d16 d17; d20 d21 d22 d23 d24 d25 d26 d27; d30 d31 d32 d33 d34 d35 d36 d37; d40 d41 d42 d43 d44 d45 d46 d47; d50 d51 d52 d53 d54 d55 d56 d57; d60 d61 d62 d63 d64 d65 d66 d67; d70 d71 d72 d73 d74 d75 d76 d77];

from = [1,5];
to = [4,8];
wd = sym('wd',[16,4]);
for i = 1:2
	s = (i-1)*4+1;
	[wd(s,:),wd(s+1,:),wd(s+2,:),wd(s+3,:),wd(s+8,:),wd(s+9,:),wd(s+10,:),wd(s+11,:)] = input_transform(dd(1,from(i):to(i)),dd(2,from(i):to(i)),dd(3,from(i):to(i)),dd(4,from(i):to(i)),dd(5,from(i):to(i)),dd(6,from(i):to(i)),dd(7,from(i):to(i)),dd(8,from(i):to(i)));
end

for i=1:2
	s = (i-1)*8+1;
	wdd1 = [wd(s,:);wd(s+1,:);wd(s+2,:);wd(s+3,:)];
	wdd1 = transpose(wdd1);
	wdd2 = [wd(s+4,:);wd(s+5,:);wd(s+6,:);wd(s+7,:)];
	wdd2 = transpose(wdd2);
	
	[wd0 wd1 wd2 wd3 wd4 wd5 wd6 wd7] = input_transform(wdd1(1,:),wdd1(2,:),wdd1(3,:),wdd1(4,:),wdd2(1,:),wdd2(2,:),wdd2(3,:),wdd2(4,:));
	k = (i-1)*4;
	for j=1:4
		transform_d(j+k,1)=wd0(j);
		transform_d(j+k,2)=wd1(j);
		transform_d(j+k,3)=wd2(j);
		transform_d(j+k,4)=wd3(j);
		transform_d(j+k,5)=wd4(j);
		transform_d(j+k,6)=wd5(j);
		transform_d(j+k,7)=wd6(j);
		transform_d(j+k,8)=wd7(j);
	end
end

fp = fopen('input_transform-2.txt','w');
for y = 1:8
	for x = 1:8
		fprintf(fp,'%s\n',transform_d(y,x));
	end
	fprintf(fp,'\n\n\n');
end
fclose(fp);

% transform output matrix.
mm = sym('m',[8,8]);

