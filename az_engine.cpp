// az_engine.cpp — Python-faithful ruleset (compilable)
// FIXES to perfectly follow your Python rules:
// 1) Implement remove_last_move() (Heidari rule hook) exactly like Python.
// 2) Game over must require totalTerr >= n2 AND stability checkGameOver().
// 3) If you play inside your OWN territory, that territory cell is cleared + count decremented (Python update_territories early-return).
// 4) removeSingleTerritory(pos): if you place a stone onto opponent territory, decrement opponent territory count (no terr-cell clear), like Python.
//
// Notes:
// - The core move legality, duplicate-state rejection, enclosure BFS, captures, and opponent-territory BFS clear are preserved.
// - The order matches Python: place stone -> duplicate test -> update_other_stones -> append move_history.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <array>
#include <future>
#include <limits>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace std;

// ============================================================
// ZOBRIST
// ============================================================

static inline uint64_t splitmix64(uint64_t &x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// ============================================================
// BOARD
// ============================================================

struct Board {

    int n = 0, n2 = 0;

    vector<uint8_t> stones;   // 0,1,2
    vector<uint8_t> terr;     // 0,1,2

    int terrCount[3] = {0,0,0};
    int totalTerr = 0;

    vector<int> moveHist;

    vector<uint64_t> zob;
    uint64_t hash = 0;
    unordered_set<uint64_t> hashSet;

    vector<array<int,4>> neigh4;

    vector<int> queueBuf;
    vector<int> bfsBuf;

    vector<uint32_t> vis;
    uint32_t visStamp = 1;

    vector<uint32_t> touchedStone, touchedTerr;
    uint32_t touchStamp = 1;

    struct Undo {
        uint64_t prevHash = 0;
        int prevTotalTerr = 0;
        int prevTerrCount1 = 0;
        int prevTerrCount2 = 0;
        size_t prevMoveHistSize = 0;

        vector<pair<int,uint8_t>> stonePrev;
        vector<pair<int,uint8_t>> terrPrev;

        bool addedHash=false;
        uint64_t addedHashValue=0;

        void clear(){ stonePrev.clear(); terrPrev.clear(); addedHash=false; addedHashValue=0; }
    };

    Board()=default;
    Board(int size){ init(size); }

    inline int idx(int x,int y) const { return y*n + x; }

    void init(int size){
        n=size; n2=n*n;
        stones.assign(n2,0);
        terr.assign(n2,0);

        terrCount[1]=terrCount[2]=0;
        totalTerr=0;

        moveHist.clear();

        zob.assign(n2*3,0);
        uint64_t seed=0x12345678abcdefULL ^ (uint64_t)size;
        for(auto &z:zob) z=splitmix64(seed);

        hash=0;
        hashSet.clear();

        neigh4.assign(n2,{-1,-1,-1,-1});
        int dx[4]={0,1,0,-1};
        int dy[4]={-1,0,1,0};

        for(int y=0;y<n;y++)for(int x=0;x<n;x++){
            int id=idx(x,y);
            for(int k=0;k<4;k++){
                int nx=x+dx[k], ny=y+dy[k];
                if(0<=nx&&nx<n&&0<=ny&&ny<n) neigh4[id][k]=idx(nx,ny);
            }
        }

        queueBuf.reserve(n2);
        bfsBuf.reserve(n2);
        vis.assign(n2,0);
        touchedStone.assign(n2,0);
        touchedTerr.assign(n2,0);
    }

    inline void markStonePrev(Undo &u,int id){
        if(touchedStone[id]!=touchStamp){
            touchedStone[id]=touchStamp;
            u.stonePrev.push_back({id,stones[id]});
        }
    }
    inline void markTerrPrev(Undo &u,int id){
        if(touchedTerr[id]!=touchStamp){
            touchedTerr[id]=touchStamp;
            u.terrPrev.push_back({id,terr[id]});
        }
    }

    inline void incTerr(uint8_t p,int d){
        terrCount[p]+=d;
        totalTerr+=d;
    }

    inline void setStoneTracked(Undo &u,int id,uint8_t v){
        uint8_t old=stones[id];
        if(old==v) return;
        markStonePrev(u,id);
        hash ^= zob[id*3+old];
        hash ^= zob[id*3+v];
        stones[id]=v;
    }

    inline void setTerrTracked(Undo &u,int id,uint8_t v){
        uint8_t old=terr[id];
        if(old==v) return;
        markTerrPrev(u,id);
        terr[id]=v;
    }

    inline void removeStoneTracked(Undo &u,int id){
        uint8_t s=stones[id];
        if(!s) return;
        // Python: removeStone decrements territory for the stone owner, then clears stone.
        incTerr(s,-1);
        setStoneTracked(u,id,0);
    }

    // Python bfs_enclosed_territory: explores region of "not player stones"
    // and aborts if touches >2 distinct walls.
    vector<int>& bfs_enclosed(uint8_t player,int start){
        bfsBuf.clear();
        queueBuf.clear();

        if(++visStamp==0){ fill(vis.begin(),vis.end(),0); visStamp=1; }

        // walls bitmask: 1=top,2=bottom,4=left,8=right
        uint8_t walls=0;

        queueBuf.push_back(start);
        vis[start]=visStamp;

        size_t qh=0;
        while(qh<queueBuf.size()){
            int id=queueBuf[qh++];
            bfsBuf.push_back(id);

            int x=id%n, y=id/n;

            if(y==0)   walls|=1;
            if(y==n-1) walls|=2;
            if(x==0)   walls|=4;
            if(x==n-1) walls|=8;

            if(__builtin_popcount((unsigned)walls)>2){
                bfsBuf.clear();     // Python returns None => treat as empty
                return bfsBuf;
            }

            for(int k=0;k<4;k++){
                int nid=neigh4[id][k];
                if(nid<0) continue;
                if(vis[nid]==visStamp) continue;
                if(stones[nid]==player) continue;
                vis[nid]=visStamp;
                queueBuf.push_back(nid);
            }
        }
        return bfsBuf;
    }

    // Python bfs_update_opponent_territory: clears connected (4-neigh) territory cells
    // that are NOT player (and not None).
    void bfs_clear_opponent_terr(Undo &u,uint8_t player,int start){
        queueBuf.clear();
        queueBuf.push_back(start);

        auto clearCell=[&](int id){
            uint8_t t=terr[id];
            if(t!=0 && t!=player){
                incTerr(t,-1);
                setTerrTracked(u,id,0);
            }
        };

        clearCell(start);

        size_t qh=0;
        while(qh<queueBuf.size()){
            int id=queueBuf[qh++];
            for(int k=0;k<4;k++){
                int nid=neigh4[id][k];
                if(nid<0) continue;
                uint8_t t=terr[nid];
                if(t!=0 && t!=player){
                    clearCell(nid);
                    queueBuf.push_back(nid);
                }
            }
        }
    }

    // Python remove_last_move(position):
    // If last move is a stone that lies inside someone else's territory, then:
    //   increment that territory owner's count +1
    //   remove the stone (which decrements stone-owner count by 1)
    // Net effect matches Python.
    bool remove_last_move(Undo &u){
        if(moveHist.empty()) return false;
        int id = moveHist.back();

        uint8_t s = stones[id];
        uint8_t t = terr[id];

        if(s!=0 && t!=0 && s!=t){
            incTerr(t, +1);
            setStoneTracked(u, id, 0);  // does NOT decrement here; Python removeStone() does decrement.
            // But Python remove_last_move() calls removeStone(), which decrements territory for stone owner.
            // We must do that too:
            incTerr(s, -1);
            return true;
        }
        return false;
    }

    // Python removeSingleTerritory(position):
    // If territory exists at this cell AND a stone exists AND they differ:
    // decrement the territory owner's count by 1.
    // (Does NOT clear the territory grid cell.)
    void remove_single_territory_count_only(uint8_t player, int pos){
        uint8_t t = terr[pos];
        uint8_t s = stones[pos];
        if(t!=0 && s!=0 && t!=s){
            incTerr(t, -1);
        }
    }

    // update_after_move is Python update_other_stones (except move_history append, which happens after)
    bool update_after_move(Undo &u,uint8_t player,int pos,
                           bool *capturedByPlacement=nullptr){
        if(capturedByPlacement) *capturedByPlacement = false;

        // Python update_territories early case:
        // If you play on a cell already marked as your territory, that territory cell is removed
        // and your territory count is decremented by 1, and no enclosure updates happen.
        if(terr[pos]==player){
            incTerr(player, -1);
            setTerrTracked(u, pos, 0);
            // still must apply removeSingleTerritory logic? In Python this early-return happens inside update_territories
            // and then update_other_stones continues to end, including removeSingleTerritory(position).
            // BUT since terr[pos]==player, removeSingleTerritory would do nothing. So we can return early safely.
            return false;
        }

        vector<int> allEnc;
        allEnc.reserve(n2);

        // Try to enclose from each 4-neigh neighbor that isn't your stone
        for(int k=0;k<4;k++){
            int nid=neigh4[pos][k];
            if(nid<0 || stones[nid]==player) continue;

            auto &enc=bfs_enclosed(player,nid);
            if(enc.empty()) continue;

            for(int c:enc){
                if(terr[c]!=player){
                    if(terr[c]!=0) incTerr(terr[c],-1);
                    incTerr(player,1);
                    setTerrTracked(u,c,player);
                }
                allEnc.push_back(c);
            }
        }

        // remove_stones_in_territory
        bool captured=false;
        for(int c:allEnc){
            uint8_t s=stones[c];
            if(s!=0 && s!=player){
                removeStoneTracked(u,c);
                captured=true;
                if(capturedByPlacement) *capturedByPlacement = true;
            }
        }

        // if captured: bfs_update_opponent_territory
        if(captured) bfs_clear_opponent_terr(u,player,pos);

        // Python:
        // if self.move_history:
        //   if remove_last_move(self.move_history[-1]): captured=True
        if(!moveHist.empty()){
            if(remove_last_move(u)){
                captured=true;
            }
        }

        // Python removeSingleTerritory(position) always at end
        remove_single_territory_count_only(player, pos);

        return captured;
    }

    int applyMoveTracked(Undo &u,uint8_t player,int pos){
        u.clear();
        u.prevHash=hash;
        u.prevTotalTerr=totalTerr;
        u.prevTerrCount1=terrCount[1];
        u.prevTerrCount2=terrCount[2];
        u.prevMoveHistSize=moveHist.size();

        if(++touchStamp==0){
            fill(touchedStone.begin(),touchedStone.end(),0);
            fill(touchedTerr.begin(),touchedTerr.end(),0);
            touchStamp=1;
        }

        if(stones[pos]!=0) return 1;

        uint8_t prevTerr = terr[pos];  // remember territory owner

        // --- place stone ---
        setStoneTracked(u,pos,player);

        if (terr[pos] != 0) {
            uint8_t t = terr[pos];
            incTerr(t, -1);
            setTerrTracked(u, pos, 0);
        }

        incTerr(player, +1);

        // superko
        if(hashSet.count(hash)){
            undoMoveTracked(u);
            return 2;
        }

        hashSet.insert(hash);
        u.addedHash=true;
        u.addedHashValue=hash;

        // 🔥 capture logic happens ONCE
        bool capturedByPlacement = false;
        update_after_move(u,player,pos,&capturedByPlacement);

        // You may invade opponent territory only if this placement
        // immediately removes an opponent stone.
        if (prevTerr != 0 && prevTerr != player && !capturedByPlacement) {
            // illegal move: undo everything
            undoMoveTracked(u);
            return 3;
        }

        moveHist.push_back(pos);
        return 0;
    }

    void undoMoveTracked(const Undo &u){
        for(auto&p:u.stonePrev) stones[p.first]=p.second;
        for(auto&p:u.terrPrev) terr[p.first]=p.second;

        terrCount[1]=u.prevTerrCount1;
        terrCount[2]=u.prevTerrCount2;
        totalTerr=u.prevTotalTerr;
        hash=u.prevHash;

        if(u.addedHash) hashSet.erase(u.addedHashValue);
        moveHist.resize(u.prevMoveHistSize);
    }

    // stability_test exactly like your Python:
    // count diagonal enemy stones around territory cell, and apply edge exception.
    bool stability(int id) const{
        uint8_t p=terr[id];
        if(!p) return true;

        int x=id%n, y=id/n;
        int c=0;

        int dx[4]={1,1,-1,-1};
        int dy[4]={1,-1,1,-1};

        for(int k=0;k<4;k++){
            int nx=x+dx[k], ny=y+dy[k];
            if((unsigned)nx<(unsigned)n && (unsigned)ny<(unsigned)n){
                int nid=ny*n+nx;
                uint8_t s=stones[nid];
                if(s!=0 && s!=p) c++;
            }
        }

        if(c>=2) return false;
        if(c==1 && (x==0||y==0||x==n-1||y==n-1)) return false;
        return true;
    }

    // Python checkGameOver: all territory cells must be stable.
    bool checkGameOver() const{
        for(int id=0;id<n2;id++)
            if(terr[id]!=0 && !stability(id)) return false;
        return true;
    }
};

// ============================================================
// GAME
// ============================================================

struct Game {
    Board board;
    int turn=0;

    Game(int size):board(size){}

    inline uint8_t player() const{ return turn==0?1:2; }

    int apply(Board::Undo &u,int pos){
        int r=board.applyMoveTracked(u,player(),pos);
        if(r==0) turn^=1;
        return r;
    }

    void undo(const Board::Undo &u){
        turn^=1;
        board.undoMoveTracked(u);
    }

    // Python: gameover only if totalTerr >= n2 AND board.checkGameOver()
    bool over() const{
        if(board.totalTerr < board.n2) return false;
        return board.checkGameOver();
    }

    int winner() const{
        if(!over()) return 0;
        int a=board.terrCount[1], b=board.terrCount[2];
        if(a>b) return 1;
        if(b>a) return 2;
        return 0;
    }
};

// ============================================================
// PY WRAPPER
// ============================================================

struct PyGame {
    Game g;
    vector<Board::Undo> undoStack;

    PyGame(int size):g(size){ undoStack.reserve(size*size*4); }


    int size()const{return g.board.n;}
    int n2()const{return g.board.n2;}
    int turn()const{return g.turn;}
    int current_player()const{return g.player();}
    bool is_over()const{return g.over();}
    int winner()const{return g.winner();}

    int score_p1() const { return g.board.terrCount[1]; }
    int score_p2() const { return g.board.terrCount[2]; }
    int total_territory() const { return g.board.totalTerr; }

    int apply(int mv){
        if(mv < 0 || mv >= g.board.n2) return 4;
        undoStack.emplace_back();
        auto &u=undoStack.back();
        int r=g.apply(u,mv);
        if(r!=0) undoStack.pop_back();
        return r;
    }

    void undo(){
        if(undoStack.empty()) return;
        g.undo(undoStack.back());
        undoStack.pop_back();
    }

    vector<int> legal_moves_all() const{
        vector<int> m;
        m.reserve(g.board.n2);
        for(int i=0;i<g.board.n2;i++){
            if(g.board.stones[i]==0)
                m.push_back(i);
        }
        return m;
    }

    vector<int> legal_moves_playable() const{
        vector<int> m;
        m.reserve(g.board.n2);
        for(int i=0;i<g.board.n2;i++){
            if(g.board.stones[i]!=0) continue;
            PyGame copy = *this;
            if(copy.apply(i)==0) m.push_back(i);
        }
        return m;
    }

    // expose zobrist hash
    uint64_t hash() const {
        return g.board.hash;
    }

    vector<uint8_t> stones() const {
        return g.board.stones;
    }

    vector<uint8_t> territories() const {
        return g.board.terr;
    }

};

// ============================================================
// ALPHA-BETA MINIMAX
// ============================================================

struct TinyValueModel {
    static constexpr int N = 7;
    static constexpr int N2 = N * N;
    static constexpr int INPUTS = 5;
    static constexpr int HIDDEN = 8;

    float scale = 500.0f;
    array<float, HIDDEN * INPUTS * 3 * 3> conv1Weight;
    array<float, HIDDEN> conv1Bias;
    array<float, HIDDEN * HIDDEN * 3 * 3> conv2Weight;
    array<float, HIDDEN> conv2Bias;
    array<float, HIDDEN * N2> outputWeight;
    float outputBias = 0.0f;

    template<size_t Size>
    static void read_array(ifstream &input, array<float, Size> &values) {
        input.read(reinterpret_cast<char *>(values.data()), sizeof(float) * Size);
        if(!input) throw runtime_error("Truncated BlitzGo value model.");
    }

    explicit TinyValueModel(const string &path) {
        ifstream input(path, ios::binary);
        if(!input) throw runtime_error("Unable to open BlitzGo value model: " + path);
        array<char, 9> magic;
        input.read(magic.data(), magic.size());
        if(!input || string(magic.data(), magic.size()) != "BLITZVAL1") {
            throw runtime_error("Invalid BlitzGo value model: " + path);
        }
        input.read(reinterpret_cast<char *>(&scale), sizeof(scale));
        if(!input) throw runtime_error("Truncated BlitzGo value model.");
        read_array(input, conv1Weight);
        read_array(input, conv1Bias);
        read_array(input, conv2Weight);
        read_array(input, conv2Bias);
        read_array(input, outputWeight);
        input.read(reinterpret_cast<char *>(&outputBias), sizeof(outputBias));
        if(!input) throw runtime_error("Truncated BlitzGo value model.");
    }

    int evaluate(const PyGame &game, int rootPlayer) const {
        if(game.size() != N) throw runtime_error("Value model expects a 7x7 board.");
        int opponent = rootPlayer == 1 ? 2 : 1;
        array<float, INPUTS * N2> board = {};
        for(int pos = 0; pos < N2; pos++) {
            uint8_t stone = game.g.board.stones[pos];
            uint8_t territory = game.g.board.terr[pos];
            board[0 * N2 + pos] = stone == rootPlayer;
            board[1 * N2 + pos] = stone == opponent;
            board[2 * N2 + pos] = territory == rootPlayer;
            board[3 * N2 + pos] = territory == opponent;
            board[4 * N2 + pos] = stone == 0;
        }

        array<float, HIDDEN * N2> hidden1 = {};
        array<float, HIDDEN * N2> hidden2 = {};
        for(int out = 0; out < HIDDEN; out++) {
            for(int y = 0; y < N; y++) for(int x = 0; x < N; x++) {
                float value = conv1Bias[out];
                for(int in = 0; in < INPUTS; in++) {
                    for(int ky = 0; ky < 3; ky++) for(int kx = 0; kx < 3; kx++) {
                        int iy = y + ky - 1, ix = x + kx - 1;
                        if(iy < 0 || iy >= N || ix < 0 || ix >= N) continue;
                        int weight = ((out * INPUTS + in) * 3 + ky) * 3 + kx;
                        value += conv1Weight[weight] * board[in * N2 + iy * N + ix];
                    }
                }
                hidden1[out * N2 + y * N + x] = max(0.0f, value);
            }
        }
        for(int out = 0; out < HIDDEN; out++) {
            for(int y = 0; y < N; y++) for(int x = 0; x < N; x++) {
                float value = conv2Bias[out];
                for(int in = 0; in < HIDDEN; in++) {
                    for(int ky = 0; ky < 3; ky++) for(int kx = 0; kx < 3; kx++) {
                        int iy = y + ky - 1, ix = x + kx - 1;
                        if(iy < 0 || iy >= N || ix < 0 || ix >= N) continue;
                        int weight = ((out * HIDDEN + in) * 3 + ky) * 3 + kx;
                        value += conv2Weight[weight] * hidden1[in * N2 + iy * N + ix];
                    }
                }
                hidden2[out * N2 + y * N + x] = max(0.0f, value);
            }
        }

        float value = outputBias;
        for(size_t i = 0; i < hidden2.size(); i++) value += outputWeight[i] * hidden2[i];
        return (int)lround(scale * tanh(value));
    }
};

static shared_ptr<const TinyValueModel> load_value_model(const string &path) {
    if(path.empty()) return nullptr;
    static mutex cacheMutex;
    static unordered_map<string, weak_ptr<const TinyValueModel>> cache;
    lock_guard<mutex> lock(cacheMutex);
    auto found = cache.find(path);
    if(found != cache.end()) {
        auto model = found->second.lock();
        if(model) return model;
    }
    auto model = make_shared<const TinyValueModel>(path);
    cache[path] = model;
    return model;
}

struct Minimax {
    uint64_t maxStates;
    uint64_t statesSearched = 0;
    int completedDepth = 0;
    int internalTopK = 0;
    shared_ptr<const TinyValueModel> valueModel;

    enum Bound : uint8_t {
        EXACT = 0,
        LOWER = 1,
        UPPER = 2,
    };

    struct TTEntry {
        int depth = -1;
        int value = 0;
        int bestMove = -1;
        Bound bound = EXACT;
    };

    unordered_map<uint64_t, TTEntry> table;

    explicit Minimax(uint64_t maxStates_=1000000, int internalTopK_=0,
                     const string &valueModelPath_="")
        : maxStates(max<uint64_t>(1, maxStates_)),
          internalTopK(max(0, internalTopK_)),
          valueModel(load_value_model(valueModelPath_)) {
        table.reserve(1 << 16);
    }

    explicit Minimax(uint64_t maxStates_, int internalTopK_,
                     shared_ptr<const TinyValueModel> valueModel_)
        : maxStates(max<uint64_t>(1, maxStates_)),
          internalTopK(max(0, internalTopK_)),
          valueModel(std::move(valueModel_)) {
        table.reserve(1 << 16);
    }

    uint64_t position_key(const PyGame &game) const {
        const Board &b = game.g.board;
        uint64_t key = b.hash ^ (uint64_t)(game.g.turn + 1) * 0x9e3779b97f4a7c15ULL;
        for(int i = 0; i < b.n2; i++) {
            key ^= (uint64_t)(b.terr[i] + 1) * (0xbf58476d1ce4e5b9ULL + (uint64_t)i);
            key = (key << 7) | (key >> 57);
        }
        return key;
    }

    int evaluate(const PyGame &game, int rootPlayer) const {
        if(valueModel) return valueModel->evaluate(game, rootPlayer);
        const Board &b = game.g.board;
        int opponent = rootPlayer == 1 ? 2 : 1;
        int value = 0;

        value += 100 * (b.terrCount[rootPlayer] - b.terrCount[opponent]);

        int rootStones = 0;
        int opponentStones = 0;
        int rootAdjEmpty = 0;
        int opponentAdjEmpty = 0;
        int rootInfluence = 0;
        int opponentInfluence = 0;
        int center = b.n / 2;

        for(int id = 0; id < b.n2; id++) {
            uint8_t stone = b.stones[id];
            if(!stone) continue;

            int x = id % b.n;
            int y = id / b.n;
            int centerBonus = b.n - (abs(x - center) + abs(y - center));

            int adjEmpty = 0;
            for(int k = 0; k < 4; k++) {
                int nid = b.neigh4[id][k];
                if(nid >= 0 && b.stones[nid] == 0) adjEmpty++;
            }

            if(stone == rootPlayer) {
                rootStones++;
                rootAdjEmpty += adjEmpty;
                rootInfluence += centerBonus;
            } else if(stone == opponent) {
                opponentStones++;
                opponentAdjEmpty += adjEmpty;
                opponentInfluence += centerBonus;
            }
        }

        value += 8 * (rootStones - opponentStones);
        value += 3 * (rootAdjEmpty - opponentAdjEmpty);
        value += 2 * (rootInfluence - opponentInfluence);
        return value;
    }

    int score_move(const PyGame &game, int move, int ttMove) const {
        if(move == ttMove) return numeric_limits<int>::max() / 4;

        const Board &b = game.g.board;
        int player = game.current_player();
        int opponent = player == 1 ? 2 : 1;
        int score = 0;

        if(move < 0 || move >= b.n2 || b.stones[move] != 0) return numeric_limits<int>::min();

        uint8_t targetTerr = b.terr[move];
        if(targetTerr == opponent) score += 600;
        else if(targetTerr == player) score += 120;

        int friendAdj = 0;
        int opponentAdj = 0;
        int emptyAdj = 0;
        for(int k = 0; k < 4; k++) {
            int nid = b.neigh4[move][k];
            if(nid < 0) continue;
            if(b.stones[nid] == player) friendAdj++;
            else if(b.stones[nid] == opponent) opponentAdj++;
            else emptyAdj++;
        }

        score += 90 * opponentAdj;
        score += 45 * friendAdj;
        score += 8 * emptyAdj;

        if(!b.moveHist.empty()) {
            int last = b.moveHist.back();
            int lx = last % b.n, ly = last / b.n;
            int x = move % b.n, y = move / b.n;
            int dist = abs(x - lx) + abs(y - ly);
            score += max(0, 6 - dist) * 25;
        }

        int center = b.n / 2;
        int x = move % b.n, y = move / b.n;
        score += b.n - (abs(x - center) + abs(y - center));
        return score;
    }

    vector<int> ordered_moves(PyGame &game, int ttMove) const {
        vector<int> moves = game.legal_moves_all();
        stable_sort(moves.begin(), moves.end(), [&](int a, int b) {
            int sa = score_move(game, a, ttMove);
            int sb = score_move(game, b, ttMove);
            if(sa != sb) return sa > sb;
            return a < b;
        });
        return moves;
    }

    bool search(PyGame &game, int depth, int alpha, int beta,
                int rootPlayer, int &value) {
        if(statesSearched >= maxStates) return false;
        statesSearched++;

        int alphaOriginal = alpha;
        int betaOriginal = beta;
        uint64_t key = position_key(game);
        int ttMove = -1;

        auto found = table.find(key);
        if(found != table.end()) {
            const TTEntry &entry = found->second;
            ttMove = entry.bestMove;
            if(entry.depth >= depth) {
                if(entry.bound == EXACT) {
                    value = entry.value;
                    return true;
                }
                if(entry.bound == LOWER) alpha = max(alpha, entry.value);
                else if(entry.bound == UPPER) beta = min(beta, entry.value);
                if(alpha >= beta) {
                    value = entry.value;
                    return true;
                }
            }
        }

        if(depth == 0 || game.is_over()) {
            value = evaluate(game, rootPlayer);
            table[key] = {depth, value, -1, EXACT};
            return true;
        }

        bool maximizing = game.current_player() == rootPlayer;
        int best = maximizing ? numeric_limits<int>::min()
                              : numeric_limits<int>::max();
        bool foundMove = false;
        int bestMove = -1;

        vector<int> moves = ordered_moves(game, ttMove);
        if(internalTopK > 0 && (int)moves.size() > internalTopK) {
            moves.resize((size_t)internalTopK);
        }

        for(int move : moves) {
            if(game.apply(move) != 0) continue;
            foundMove = true;

            int childValue = 0;
            bool completed = search(game, depth - 1, alpha, beta,
                                    rootPlayer, childValue);
            game.undo();
            if(!completed) return false;

            if(maximizing) {
                if(childValue > best) {
                    best = childValue;
                    bestMove = move;
                }
                alpha = max(alpha, best);
            } else {
                if(childValue < best) {
                    best = childValue;
                    bestMove = move;
                }
                beta = min(beta, best);
            }

            if(beta <= alpha) break;
        }

        value = foundMove ? best : evaluate(game, rootPlayer);
        Bound bound = EXACT;
        if(value <= alphaOriginal) bound = UPPER;
        else if(value >= betaOriginal) bound = LOWER;
        table[key] = {depth, value, bestMove, bound};
        return true;
    }

    bool bestMoveAtDepth(PyGame &game, int depth, int &bestMove) {
        return bestMoveAtDepth(game, depth, game.legal_moves_all(), bestMove);
    }

    vector<int> orderedRootMoves(PyGame &game, const vector<int> &preferred) {
        vector<int> moves;
        vector<uint8_t> seen(game.n2(), 0);
        moves.reserve(game.n2());

        for(int move : preferred) {
            if(move < 0 || move >= game.n2() || seen[move]) continue;
            seen[move] = 1;
            moves.push_back(move);
        }
        for(int move : game.legal_moves_all()) {
            if(seen[move]) continue;
            seen[move] = 1;
            moves.push_back(move);
        }
        return moves;
    }

    vector<int> withMoveFirst(const vector<int> &moves, int firstMove) const {
        if(firstMove < 0) return moves;
        vector<int> ordered;
        ordered.reserve(moves.size());
        ordered.push_back(firstMove);
        for(int move : moves) {
            if(move != firstMove) ordered.push_back(move);
        }
        return ordered;
    }

    bool bestMoveAtDepth(PyGame &game, int depth, const vector<int> &rootMoves,
                         int &bestMove) {
        int rootPlayer = game.current_player();
        int bestValue = numeric_limits<int>::min();
        int alpha = numeric_limits<int>::min();
        int beta = numeric_limits<int>::max();
        bestMove = -1;

        for(int move : rootMoves) {
            if(game.apply(move) != 0) continue;

            int value = 0;
            bool completed = search(game, depth - 1, alpha, beta,
                                    rootPlayer, value);
            game.undo();
            if(!completed) return false;

            if(bestMove == -1 || value > bestValue) {
                bestValue = value;
                bestMove = move;
            }
            alpha = max(alpha, bestValue);
        }

        return true;
    }

    int best_move_ordered(PyGame &game, const vector<int> &preferred) {
        statesSearched = 0;
        completedDepth = 0;
        int bestMove = -1;
        vector<int> rootMoves = orderedRootMoves(game, preferred);

        for(int depth = 1; statesSearched < maxStates; depth++) {
            int candidate = -1;
            vector<int> depthRootMoves = withMoveFirst(rootMoves, bestMove);
            if(!bestMoveAtDepth(game, depth, depthRootMoves, candidate)) break;
            if(candidate < 0) break;
            bestMove = candidate;
            completedDepth = depth;
        }

        if(bestMove >= 0) return bestMove;

        for(int move : game.legal_moves_all()) {
            if(game.apply(move) == 0) {
                game.undo();
                return move;
            }
        }
        return -1;
    }

    int best_move_subset(PyGame &game, const vector<int> &rootMoves) {
        statesSearched = 0;
        completedDepth = 0;
        int bestMove = -1;

        for(int depth = 1; statesSearched < maxStates; depth++) {
            int candidate = -1;
            vector<int> depthRootMoves = withMoveFirst(rootMoves, bestMove);
            if(!bestMoveAtDepth(game, depth, depthRootMoves, candidate)) break;
            if(candidate < 0) break;
            bestMove = candidate;
            completedDepth = depth;
        }

        if(bestMove >= 0) return bestMove;

        for(int move : rootMoves) {
            if(game.apply(move) == 0) {
                game.undo();
                return move;
            }
        }
        return -1;
    }

    int best_move_subset_parallel(PyGame &game, const vector<int> &rootMoves,
                                  int workers) {
        statesSearched = 0;
        completedDepth = 0;
        int bestMove = -1;

        vector<int> legalRootMoves;
        vector<uint8_t> seen(game.n2(), 0);
        legalRootMoves.reserve(rootMoves.size());
        for(int move : rootMoves) {
            if(move < 0 || move >= game.n2() || seen[move]) continue;
            PyGame copy = game;
            if(copy.apply(move) != 0) continue;
            seen[move] = 1;
            legalRootMoves.push_back(move);
        }
        if(legalRootMoves.empty()) return -1;
        if(legalRootMoves.size() == 1) {
            statesSearched = 0;
            completedDepth = 0;
            return legalRootMoves.front();
        }

        workers = max(1, min(workers, (int)legalRootMoves.size()));

        struct RootResult {
            int move = -1;
            int value = numeric_limits<int>::min();
            uint64_t states = 0;
            bool completed = false;
        };

        int rootPlayer = game.current_player();
        for(int depth = 1; statesSearched < maxStates; depth++) {
            vector<int> depthRootMoves = withMoveFirst(legalRootMoves, bestMove);
            uint64_t remaining = maxStates - statesSearched;
            uint64_t perMoveBudget = max<uint64_t>(
                1,
                remaining / (uint64_t)depthRootMoves.size()
            );

            vector<RootResult> results;
            results.reserve(depthRootMoves.size());
            uint64_t depthStates = 0;
            bool completedDepthThisRound = true;
            int sharedAlpha = numeric_limits<int>::min();
            int childInternalTopK = internalTopK;

            RootResult firstResult;
            firstResult.move = depthRootMoves.front();
            {
                PyGame copy = game;
                if(copy.apply(firstResult.move) != 0) {
                    firstResult.completed = false;
                } else {
                    Minimax firstSearch(perMoveBudget, internalTopK, valueModel);
                    int value = 0;
                    bool completed = firstSearch.search(
                        copy,
                        depth - 1,
                        numeric_limits<int>::min(),
                        numeric_limits<int>::max(),
                        rootPlayer,
                        value
                    );
                    firstResult.value = value;
                    firstResult.states = firstSearch.statesSearched;
                    firstResult.completed = completed;
                }
            }
            depthStates += firstResult.states;
            completedDepthThisRound = firstResult.completed;
            if(firstResult.completed) {
                sharedAlpha = firstResult.value;
            }
            results.push_back(firstResult);

            for(size_t start = 1; start < depthRootMoves.size();
                start += (size_t)workers) {
                vector<future<RootResult>> futures;
                size_t end = min(depthRootMoves.size(), start + (size_t)workers);
                futures.reserve(end - start);

                for(size_t i = start; i < end; i++) {
                    int move = depthRootMoves[i];
                    futures.push_back(async(
                        launch::async,
                        [&game, move, depth, rootPlayer, perMoveBudget, sharedAlpha,
                         childInternalTopK, model=valueModel]() {
                            RootResult result;
                            result.move = move;

                            PyGame copy = game;
                            if(copy.apply(move) != 0) {
                                result.completed = false;
                                return result;
                            }

                            Minimax localSearch(perMoveBudget, childInternalTopK, model);
                            int value = 0;
                            bool completed = localSearch.search(
                                copy,
                                depth - 1,
                                sharedAlpha,
                                numeric_limits<int>::max(),
                                rootPlayer,
                                value
                            );

                            result.value = value;
                            result.states = localSearch.statesSearched;
                            result.completed = completed;
                            return result;
                        }
                    ));
                }

                for(auto &future : futures) {
                    RootResult result = future.get();
                    depthStates += result.states;
                    if(!result.completed) completedDepthThisRound = false;
                    results.push_back(result);
                }
            }

            if(depthStates == 0) {
                break;
            }
            if(!completedDepthThisRound || statesSearched + depthStates > maxStates) {
                statesSearched = min(maxStates, statesSearched + depthStates);
                break;
            }
            statesSearched += depthStates;

            int bestValue = numeric_limits<int>::min();
            int candidate = -1;
            for(const RootResult &result : results) {
                if(result.move < 0) continue;
                if(candidate == -1 || result.value > bestValue) {
                    bestValue = result.value;
                    candidate = result.move;
                }
            }

            if(candidate < 0) break;
            bestMove = candidate;
            completedDepth = depth;
        }

        if(bestMove >= 0) return bestMove;
        return legalRootMoves.front();
    }

    py::dict best_move_subset_parallel_info(PyGame &game,
                                            const vector<int> &rootMoves,
                                            int workers) {
        statesSearched = 0;
        completedDepth = 0;
        int bestMove = -1;

        vector<int> legalRootMoves;
        vector<uint8_t> seen(game.n2(), 0);
        legalRootMoves.reserve(rootMoves.size());
        for(int move : rootMoves) {
            if(move < 0 || move >= game.n2() || seen[move]) continue;
            PyGame copy = game;
            if(copy.apply(move) != 0) continue;
            seen[move] = 1;
            legalRootMoves.push_back(move);
        }

        struct RootResult {
            int move = -1;
            int value = numeric_limits<int>::min();
            uint64_t states = 0;
            bool completed = false;
        };

        vector<RootResult> bestResults;
        if(legalRootMoves.empty()) {
            py::dict result;
            result["best_move"] = -1;
            result["moves"] = vector<int>{};
            result["scores"] = vector<int>{};
            result["states"] = vector<uint64_t>{};
            result["states_searched"] = statesSearched;
            result["completed_depth"] = completedDepth;
            return result;
        }
        if(legalRootMoves.size() == 1) {
            bestMove = legalRootMoves.front();
            bestResults.push_back({bestMove, 0, 0, true});
        } else {
            workers = max(1, min(workers, (int)legalRootMoves.size()));

            int rootPlayer = game.current_player();
            for(int depth = 1; statesSearched < maxStates; depth++) {
                vector<int> depthRootMoves = withMoveFirst(legalRootMoves, bestMove);
                uint64_t remaining = maxStates - statesSearched;
                uint64_t perMoveBudget = max<uint64_t>(
                    1,
                    remaining / (uint64_t)depthRootMoves.size()
                );

                vector<RootResult> results;
                results.reserve(depthRootMoves.size());
                uint64_t depthStates = 0;
                bool completedDepthThisRound = true;
                int sharedAlpha = numeric_limits<int>::min();
                int childInternalTopK = internalTopK;

                RootResult firstResult;
                firstResult.move = depthRootMoves.front();
                {
                    PyGame copy = game;
                    if(copy.apply(firstResult.move) != 0) {
                        firstResult.completed = false;
                    } else {
                        Minimax firstSearch(perMoveBudget, internalTopK, valueModel);
                        int value = 0;
                        bool completed = firstSearch.search(
                            copy,
                            depth - 1,
                            numeric_limits<int>::min(),
                            numeric_limits<int>::max(),
                            rootPlayer,
                            value
                        );
                        firstResult.value = value;
                        firstResult.states = firstSearch.statesSearched;
                        firstResult.completed = completed;
                    }
                }
                depthStates += firstResult.states;
                completedDepthThisRound = firstResult.completed;
                if(firstResult.completed) {
                    sharedAlpha = firstResult.value;
                }
                results.push_back(firstResult);

                for(size_t start = 1; start < depthRootMoves.size();
                    start += (size_t)workers) {
                    vector<future<RootResult>> futures;
                    size_t end = min(depthRootMoves.size(), start + (size_t)workers);
                    futures.reserve(end - start);

                    for(size_t i = start; i < end; i++) {
                        int move = depthRootMoves[i];
                        futures.push_back(async(
                            launch::async,
                            [&game, move, depth, rootPlayer, perMoveBudget, sharedAlpha,
                             childInternalTopK, model=valueModel]() {
                                RootResult result;
                                result.move = move;

                                PyGame copy = game;
                                if(copy.apply(move) != 0) {
                                    result.completed = false;
                                    return result;
                                }

                                Minimax localSearch(perMoveBudget, childInternalTopK, model);
                                int value = 0;
                                bool completed = localSearch.search(
                                    copy,
                                    depth - 1,
                                    sharedAlpha,
                                    numeric_limits<int>::max(),
                                    rootPlayer,
                                    value
                                );

                                result.value = value;
                                result.states = localSearch.statesSearched;
                                result.completed = completed;
                                return result;
                            }
                        ));
                    }

                    for(auto &future : futures) {
                        RootResult result = future.get();
                        depthStates += result.states;
                        if(!result.completed) completedDepthThisRound = false;
                        results.push_back(result);
                    }
                }

                if(depthStates == 0) break;
                if(!completedDepthThisRound || statesSearched + depthStates > maxStates) {
                    statesSearched = min(maxStates, statesSearched + depthStates);
                    break;
                }
                statesSearched += depthStates;

                int bestValue = numeric_limits<int>::min();
                int candidate = -1;
                for(const RootResult &result : results) {
                    if(result.move < 0) continue;
                    if(candidate == -1 || result.value > bestValue) {
                        bestValue = result.value;
                        candidate = result.move;
                    }
                }

                if(candidate < 0) break;
                bestMove = candidate;
                bestResults = std::move(results);
                completedDepth = depth;
            }
        }

        if(bestMove < 0) bestMove = legalRootMoves.front();

        stable_sort(bestResults.begin(), bestResults.end(),
            [bestMove](const RootResult &a, const RootResult &b) {
                if(a.value != b.value) return a.value > b.value;
                if(a.move == bestMove && b.move != bestMove) return true;
                if(b.move == bestMove && a.move != bestMove) return false;
                return a.move < b.move;
            }
        );

        vector<int> moves;
        vector<int> scores;
        vector<uint64_t> states;
        moves.reserve(bestResults.size());
        scores.reserve(bestResults.size());
        states.reserve(bestResults.size());
        for(const RootResult &result : bestResults) {
            if(result.move < 0) continue;
            moves.push_back(result.move);
            scores.push_back(result.value);
            states.push_back(result.states);
        }

        py::dict result;
        result["best_move"] = bestMove;
        result["moves"] = moves;
        result["scores"] = scores;
        result["states"] = states;
        result["states_searched"] = statesSearched;
        result["completed_depth"] = completedDepth;
        return result;
    }

    int best_move(PyGame &game) {
        return best_move_ordered(game, {});
    }

    bool rootScoresAtDepth(PyGame &game, int depth, vector<int> &moves,
                           vector<int> &scores) {
        int rootPlayer = game.current_player();
        moves.clear();
        scores.clear();

        for(int move : game.legal_moves_all()) {
            if(game.apply(move) != 0) continue;

            int value = 0;
            bool completed = search(game, depth - 1,
                                    numeric_limits<int>::min(),
                                    numeric_limits<int>::max(),
                                    rootPlayer, value);
            game.undo();
            if(!completed) return false;

            moves.push_back(move);
            scores.push_back(value);
        }
        return true;
    }

    bool rootScoresAtDepthParallel(PyGame &game, int depth, int workers,
                                   vector<int> &moves, vector<int> &scores,
                                   uint64_t &depthStates) {
        vector<int> legalRootMoves = game.legal_moves_all();
        moves.clear();
        scores.clear();
        depthStates = 0;
        if(legalRootMoves.empty()) return true;

        workers = max(1, min(workers, (int)legalRootMoves.size()));
        uint64_t remaining = maxStates > statesSearched
            ? maxStates - statesSearched
            : 0;
        if(remaining == 0) return false;

        uint64_t perMoveBudget = max<uint64_t>(
            1,
            remaining / (uint64_t)legalRootMoves.size()
        );

        struct RootScore {
            int move = -1;
            int score = 0;
            uint64_t states = 0;
            bool legal = false;
            bool completed = false;
        };

        int rootPlayer = game.current_player();
        vector<RootScore> results;
        results.reserve(legalRootMoves.size());
        bool completedDepthThisRound = true;
        int childInternalTopK = internalTopK;

        for(size_t start = 0; start < legalRootMoves.size();
            start += (size_t)workers) {
            vector<future<RootScore>> futures;
            size_t end = min(legalRootMoves.size(), start + (size_t)workers);
            futures.reserve(end - start);

            for(size_t i = start; i < end; i++) {
                int move = legalRootMoves[i];
                futures.push_back(async(
                    launch::async,
                    [&game, move, depth, rootPlayer, perMoveBudget,
                     childInternalTopK, model=valueModel]() {
                        RootScore result;
                        result.move = move;

                        PyGame copy = game;
                        if(copy.apply(move) != 0) {
                            result.completed = true;
                            return result;
                        }
                        result.legal = true;

                        Minimax localSearch(perMoveBudget, childInternalTopK, model);
                        int value = 0;
                        bool completed = localSearch.search(
                            copy,
                            depth - 1,
                            numeric_limits<int>::min(),
                            numeric_limits<int>::max(),
                            rootPlayer,
                            value
                        );

                        result.score = value;
                        result.states = localSearch.statesSearched;
                        result.completed = completed;
                        return result;
                    }
                ));
            }

            for(auto &future : futures) {
                RootScore result = future.get();
                depthStates += result.states;
                if(!result.completed) completedDepthThisRound = false;
                results.push_back(result);
            }
        }

        if(!completedDepthThisRound) return false;
        if(depthStates == 0 && !results.empty()) return false;
        if(statesSearched + depthStates > maxStates) return false;

        for(const RootScore &result : results) {
            if(!result.legal) continue;
            moves.push_back(result.move);
            scores.push_back(result.score);
        }
        return true;
    }

    py::dict rank_root_moves(PyGame &game) {
        statesSearched = 0;
        completedDepth = 0;
        vector<int> bestMoves;
        vector<int> bestScores;

        for(int depth = 1; statesSearched < maxStates; depth++) {
            vector<int> moves;
            vector<int> scores;
            if(!rootScoresAtDepth(game, depth, moves, scores)) break;
            if(moves.empty()) break;
            bestMoves = std::move(moves);
            bestScores = std::move(scores);
            completedDepth = depth;
        }

        py::dict result;
        result["moves"] = bestMoves;
        result["scores"] = bestScores;
        result["states_searched"] = statesSearched;
        result["completed_depth"] = completedDepth;
        return result;
    }

    py::dict rank_root_moves_parallel(PyGame &game, int workers) {
        statesSearched = 0;
        completedDepth = 0;
        vector<int> bestMoves;
        vector<int> bestScores;

        workers = max(1, workers);
        for(int depth = 1; statesSearched < maxStates; depth++) {
            vector<int> moves;
            vector<int> scores;
            uint64_t depthStates = 0;
            if(!rootScoresAtDepthParallel(game, depth, workers, moves, scores,
                                          depthStates)) {
                statesSearched = min(maxStates, statesSearched + depthStates);
                break;
            }
            if(moves.empty()) break;
            statesSearched += depthStates;
            bestMoves = std::move(moves);
            bestScores = std::move(scores);
            completedDepth = depth;
        }

        py::dict result;
        result["moves"] = bestMoves;
        result["scores"] = bestScores;
        result["states_searched"] = statesSearched;
        result["completed_depth"] = completedDepth;
        return result;
    }

    uint64_t states_searched() const {
        return statesSearched;
    }

    int completed_depth() const {
        return completedDepth;
    }
};


PYBIND11_MODULE(az_engine, m) {
    py::class_<PyGame>(m, "Game")
        .def(py::init<int>())
        .def("size", &PyGame::size)
        .def("n2", &PyGame::n2)
        .def("turn", &PyGame::turn)
        .def("current_player", &PyGame::current_player)
        .def("is_over", &PyGame::is_over)
        .def("winner", &PyGame::winner)
        .def("score_p1", &PyGame::score_p1)
        .def("score_p2", &PyGame::score_p2)
        .def("total_territory", &PyGame::total_territory)
        .def("apply", &PyGame::apply)
        .def("undo", &PyGame::undo)
        .def("legal_moves_all", &PyGame::legal_moves_all)
        .def("legal_moves_playable", &PyGame::legal_moves_playable)
        .def("hash", &PyGame::hash)
        .def("stones", &PyGame::stones)
        .def("territories", &PyGame::territories);

    py::class_<Minimax>(m, "Minimax")
        .def(py::init<uint64_t, int, const string &>(),
             py::arg("max_states")=1000000,
             py::arg("internal_top_k")=0,
             py::arg("value_model")="")
        .def("evaluate", &Minimax::evaluate,
             py::arg("game"),
             py::arg("root_player"))
        .def("best_move", &Minimax::best_move,
             py::arg("game"))
        .def("best_move_ordered", &Minimax::best_move_ordered,
             py::arg("game"),
             py::arg("preferred_moves"))
        .def("best_move_subset", &Minimax::best_move_subset,
             py::arg("game"),
             py::arg("root_moves"))
        .def("best_move_subset_parallel", &Minimax::best_move_subset_parallel,
             py::arg("game"),
             py::arg("root_moves"),
             py::arg("workers")=0,
             py::call_guard<py::gil_scoped_release>())
        .def("best_move_subset_parallel_info",
             &Minimax::best_move_subset_parallel_info,
             py::arg("game"),
             py::arg("root_moves"),
             py::arg("workers")=0)
        .def("rank_root_moves", &Minimax::rank_root_moves,
             py::arg("game"))
        .def("rank_root_moves_parallel", &Minimax::rank_root_moves_parallel,
             py::arg("game"),
             py::arg("workers")=10)
        .def("states_searched", &Minimax::states_searched)
        .def("completed_depth", &Minimax::completed_depth);
}
