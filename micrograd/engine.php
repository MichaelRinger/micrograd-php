<?php

class Value
{
    public function __construct(float $data, array $_prev = [], ?string $_op = null)
    {
        $this->data = $data;
        $this->grad = 0;

        $this->_prev = $_prev;
        $this->_op = $_op;
        $this->_backward = fn () => null;
    }

    public function add($other)
    {
        $other = !$other instanceof Value ? new Value($other) : $other;
        # f(x) = x + o
        $out = new Value($this->data + $other->data, [$this, $other], '+');

        # f'(x) = 1
        $out->_backward = function () use ($other, $out) {
            $this->grad += $out->grad;
            $other->grad += $out->grad;
        };
        return $out;
    }

    public function sub($other)
    {
        $other = !$other instanceof Value ? new Value($other) : $other;
        return $this->add($other->mul(-1));
    }

    public function mul($other)
    {
        $other = !$other instanceof Value ? new Value($other) : $other;
        # f(x) = x * o
        $out = new Value($this->data * $other->data, [$this, $other], '*');

        # f'(x) = o
        $out->_backward = function () use ($other, $out) {
            $this->grad += $other->data * $out->grad;
            $other->grad += $this->data * $out->grad;
        };
        return $out;
    }

    public function pow($other)
    {
        assert(in_array(gettype($other), ['integer', 'double']), 'only supporting int/float powers for now');
        # f(x) = x**o
        $out = new Value($this->data ** $other, [$this], "**{$other}");

        # f'(x) = o*x**(o-1)
        $out->_backward = function () use ($other, $out) {
            $this->grad += ($other * $this->data ** ($other - 1)) * $out->grad;
        };
        return $out;
    }

    public function relu()
    {
        $out = new Value($this->data > 0 ? $this->data : 0, [$this], 'ReLU');

        $out->_backward = function () use ($out) {
            $this->grad += ($out->data > 0) * $out->grad;
        };
        return $out;
    }

    public function exp()
    {
        # f(x) = e**x
        $out = new Value(exp($this->data), [$this], 'ReLU');

        # f'(x) = e**x * x'
        $out->_backward = function () use ($out) {
            $this->grad += $out->data * $out->grad;
        };
        return $out;
    }

    public function tanh()
    {
        $x = $this->data;
        $t = (exp(2 * $x) - 1) / (exp(2 * $x) + 1);
        $out = new Value($t, [$this], 'tanh');

        $out->_backward = function () use ($out, $t) {
            $this->grad += (1 - $t ** 2) * $out->grad;
        };
        return $out;
    }

    public function div($other)
    {
        return $this->mul($other->pow(-1));
    }

    public function backward()
    {
        global $topo;
        $topo = [];
        $buildTopo = function ($v) use (&$buildTopo) {
            global $topo;
            if ($v instanceof Value) {
                $topo[] = $v;
                foreach ($v->_prev as $v) {
                    $buildTopo($v);
                }
            }
        };
        $buildTopo($this);
        $this->grad = 1;
        foreach ($topo as $v) {
            ($v->_backward)();
        }
    }
}
