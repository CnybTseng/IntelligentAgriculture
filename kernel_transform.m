function [w0 w1 w2 w3 w4 w5 w6 w7] = kernel_transform(g0,g1,g2)

const_4 = 4;
w2 = g0 + g2;
w4 = g0 + const_4 * g2;
w6 = g2 + const_4 * g0;

two_g1 = g1 * 2;
w1 = w2 + g1;
w2 = w2 - g1;
w3 = w4 + two_g1;
w4 = w4 - two_g1;
w5 = w6 + two_g1;
w6 = w6 - two_g1;

w1 = w1 * (-2/9);
w2 = w2 * (-2/9);
w3 = w3 / 90;
w4 = w4 / 90;
w5 = w5 / 180;
w6 = w6 / 180;

w0 = g0;
w7 = g2;