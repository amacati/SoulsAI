@0x9f075f591ea0d034;

struct DQNSample {
  obs @0 :List(List(List(UInt8)));
  action @1 :Int32;
  reward @2 :Float32;
  nextObs @3 :List(List(List(UInt8)));
  terminated @4 :Bool;
  truncated @5 :Bool;
  info @6 :Info;
  epSteps @7 :Int32;
  modelId @8 :Text;
}

struct Info {
}

struct EpisodeInfo {
  epReward @0 :Float32;
  epSteps @1 :Int32;
  bossHp @2 :Float32;
  win @3 :Bool;
  eps @4 :Float32;
  modelId @5 :Text;
}

struct Telemetry {
  epReward @0 :Float32;
  epSteps @1 :Int32;
  totalSteps @2 :Int32;
  bossHp @3 :Float32;
  win @4 :Bool;
  eps @5 :Float32;
}
