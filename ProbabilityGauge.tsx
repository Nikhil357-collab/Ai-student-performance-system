"use client";

import GaugeComponent from "react-gauge-component";

interface Props {
  value: number;
}

export default function ProbabilityGauge({ value }: Props) {

  return (
    <div className="w-full flex justify-center">

      <GaugeComponent
        value={value * 100}
        type="radial"
        labels={{
          valueLabel: {
            formatTextValue: value => `${value.toFixed(1)}%`
          }
        }}
      />

    </div>
  );
}