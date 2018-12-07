function [t0 t1 t2 t3 t4 t5 t6 t7] = input_transform(d0,d1,d2,d3,d4,d5,d6,d7)

wd0 = d0 - d6;
d4_sub_d2 = d4 - d2;
wd7 = d7 - d1;
d3_sub_d5 = d3 - d5;
wd1 = d2 + d6;
wd2 = d1 + d5;
wd4 = d5 + 0.25 * d1;
wd5 = d6 - 5 * d4;
wd3 = d6 + 0.25 * d2;
wd6 = d1 + 0.25 * d5;

wd0 = wd0 + 5.25 * d4_sub_d2;
wd7 = wd7 + 5.25 * d3_sub_d5;

wd1 = wd1 - 4.25 * d4;
wd2 = wd2 - 4.25 * d3;

wd3 = wd3 - 1.25 * d4;
wd5 = wd5 + 4.0 * d2;
wd4 = wd4 - 1.25 * d3;
wd6 = wd6 - 1.25 * d3;

t0 = wd0;
t1 = wd1 + wd2;
t2 = wd1 - wd2;
t3 = wd3 + wd4 * 2;
t4 = wd3 - wd4 * 2;
t5 = wd5 + wd6 * 2;
t6 = wd5 - wd6 * 2;
t7 = wd7;