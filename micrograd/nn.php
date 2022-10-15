<?php

class Module
{
    public function zeroGrad()
    {
        foreach ($this->parameters() as $p) {
            $p->grad = 0;
        }
    }

    public function parameters()
    {
        return [];
    }
}

class Neuron extends Module
{
    public function __construct(int $nin, bool $nonlin = True)
    {
        $this->w = array_map(fn () => new Value(rand(-10 ** 14, 10 ** 14) / 10 ** 14), range(1, $nin));
        $this->b = new Value(0);
        $this->nonlin = $nonlin;
    }

    public function forward($x)
    {
        assert(count($x) == count($this->w), 'input lenghts doesnt match.');
        $act = array_reduce(array_keys($x), fn ($act, $i) => $act->add($this->w[$i]->mul($x[$i])), $this->b);
        return $this->nonlin ? $act->relu() : $act;
    }

    public function parameters()
    {
        return array_merge($this->w, [$this->b]);
    }
}

class Layer extends Module
{
    public function __construct(int $nin, int $nout, bool $nonlin = True)
    {
        $this->neurons = array_map(fn () => new Neuron($nin, $nonlin), range(1, $nout));
    }

    public function forward($x)
    {
        $out = array_map(fn ($n) => $n->forward($x), $this->neurons);
        return $out;
    }

    public function parameters()
    {
        return array_merge(...array_map(fn ($n) => $n->parameters(), $this->neurons));
    }
}

class MLP extends Module
{
    public function __construct(int $nin, array $nouts)
    {
        $sz = array_merge([$nin], $nouts);
        $this->layers = array_map(fn ($i) => new Layer($sz[$i], $sz[$i + 1], ($i + 1) != count($nouts)), array_keys($nouts));
    }

    public function forward($x)
    {
        $out = array_reduce($this->layers, fn ($x, $l) => $l->forward($x), $x);
        return count($out) > 1 ? $out : $out[0];
    }

    public function parameters()
    {
        return array_merge(...array_map(fn ($l) => $l->parameters(), $this->layers));
    }
}
