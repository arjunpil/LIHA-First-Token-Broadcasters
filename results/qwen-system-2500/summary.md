# qwen-system

| setting | head | accuracy | non-English acc | c->w | w->c |
|---|---|---|---|---|---|
| default | base | 0.922 | 0.902 | 0.000 | 0.000 |
| default | head:L22H6 | 0.422 | 0.278 | 0.500 | 0.001 |
| default | head:L17H7 | 0.597 | 0.496 | 0.325 | 0.001 |
| default | head:L17H8 | 0.687 | 0.609 | 0.235 | 0.000 |
| default | head:L0H6 | 0.750 | 0.690 | 0.193 | 0.021 |
| default | head:L25H2 | 0.921 | 0.901 | 0.002 | 0.002 |
| default | head:L26H8 | 0.929 | 0.911 | 0.002 | 0.010 |
| default | head:L11H10 | 0.963 | 0.954 | 0.002 | 0.043 |
| no-system | base | 0.985 | 0.983 | 0.000 | 0.000 |
| no-system | head:L22H6 | 0.727 | 0.660 | 0.258 | 0.000 |
| no-system | head:L17H7 | 0.831 | 0.790 | 0.155 | 0.001 |
| no-system | head:L17H8 | 0.915 | 0.895 | 0.072 | 0.002 |
| no-system | head:L0H6 | 0.952 | 0.941 | 0.046 | 0.012 |
| no-system | head:L25H2 | 0.984 | 0.981 | 0.002 | 0.000 |
| no-system | head:L26H8 | 0.986 | 0.984 | 0.000 | 0.001 |
| no-system | head:L11H10 | 0.974 | 0.969 | 0.015 | 0.004 |
| native-system | base | 0.997 | 0.996 | 0.000 | 0.000 |
| native-system | head:L22H6 | 0.619 | 0.523 | 0.379 | 0.001 |
| native-system | head:L17H7 | 0.913 | 0.891 | 0.085 | 0.001 |
| native-system | head:L17H8 | 0.977 | 0.971 | 0.020 | 0.000 |
| native-system | head:L0H6 | 0.968 | 0.962 | 0.031 | 0.002 |
| native-system | head:L25H2 | 0.997 | 0.997 | 0.000 | 0.000 |
| native-system | head:L26H8 | 0.997 | 0.997 | 0.000 | 0.000 |
| native-system | head:L11H10 | 0.998 | 0.998 | 0.001 | 0.002 |
