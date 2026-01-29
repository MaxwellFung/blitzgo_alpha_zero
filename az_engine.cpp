// az_engine.cpp — Python-faithful ruleset (compilable)
// FIXES to perfectly follow your Python rules:
// 1) Implement remove_last_move() (Heidari rule hook) exactly like Python.
// 2) Game over must require totalTerr >= n2 AND stability checkGameOver().
// 3) If you play inside your OWN territory, that territory cell is cleared + count decremented (Python update_territories early-return).
// 4) removeSingleTerritory(pos): if you place a stone onto opponent territory, decrement opponent territory count (no terr-cell clear), like Python.
// 5) legal_moves_top20 uses 8-neighborhood like Python generateValidMoves().
//
// Notes:
// - The core move legality, duplicate-state rejection, enclosure BFS, captures, and opponent-territory BFS clear are preserved.
// - The order matches Python: place stone -> duplicate test -> update_other_stones -> append move_history.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/gil.h>
#include <pybind11/functional.h>

#include <vector>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cstdint>
#include <array>
#include <cmath>

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
    bool update_after_move(Undo &u,uint8_t player,int pos){

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

        // Python isValidMove: must be empty
        if(stones[pos]!=0) return 1;

        // Place stone
        setStoneTracked(u,pos,player);

        // Python update_other_stones begins with incrementTerritory(player, 1)
        // So we do it immediately after placement.
        incTerr(player, +1);

        // Python duplicate move check happens after placing stone and before updates.
        if(hashSet.count(hash)){
            // revert everything
            for(auto&p:u.stonePrev) stones[p.first]=p.second;
            hash=u.prevHash;
            terrCount[1]=u.prevTerrCount1;
            terrCount[2]=u.prevTerrCount2;
            totalTerr=u.prevTotalTerr;
            for(auto&p:u.terrPrev) terr[p.first]=p.second;
            moveHist.resize(u.prevMoveHistSize);
            return 2;
        }

        // record this position in superko set
        hashSet.insert(hash);
        u.addedHash=true;
        u.addedHashValue=hash;

        // update_other_stones
        update_after_move(u,player,pos);

        // Python: move_history append happens at end of placeStone()
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

    int apply(int mv){
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
        for(int i=0;i<g.board.n2;i++) if(g.board.stones[i]==0) m.push_back(i);
        return m;
    }

    // expose zobrist hash
    uint64_t hash() const {
        return g.board.hash;
    }

    // Python generateValidMoves(): top moves adjacent to existing stones, scored by 8-direction adjacency, top 20.
    vector<int> legal_moves_top20() const {
        const Board &b = g.board;
        vector<int> score(b.n2, 0);

        // 8 directions in flattened indexing (need x/y checks)
        static const int dx8[8] = {-1,0,1,-1,1,-1,0,1};
        static const int dy8[8] = {-1,-1,-1,0,0,1,1,1};

        for (int id = 0; id < b.n2; id++) if (b.stones[id]) {
            int x = id % b.n, y = id / b.n;
            for (int k=0;k<8;k++){
                int nx=x+dx8[k], ny=y+dy8[k];
                if((unsigned)nx<(unsigned)b.n && (unsigned)ny<(unsigned)b.n){
                    int nid = ny*b.n + nx;
                    if(b.stones[nid]==0) score[nid] += 1;
                }
            }
        }

        vector<pair<int,int>> cand;
        cand.reserve(b.n2);
        for (int i = 0; i < b.n2; i++)
            if (b.stones[i] == 0 && score[i] > 0)
                cand.push_back({-score[i], i});

        sort(cand.begin(), cand.end());

        vector<int> out;
        for (int i = 0; i < (int)cand.size() && i < 20; i++)
            out.push_back(cand[i].second);

        // if no adjacent moves, fallback to first 20 empties
        if (out.empty()) {
            for (int i = 0; i < b.n2 && (int)out.size() < 20; i++)
                if (b.stones[i] == 0) out.push_back(i);
        }

        return out;
    }

    py::array_t<uint8_t> state_tensor() const{
        int N=g.board.n;
        py::array_t<uint8_t> out({5,N,N});
        auto r=out.mutable_data();
        int S=N*N;
        for(int i=0;i<S;i++){
            r[i]=g.board.stones[i]==1;
            r[S+i]=g.board.stones[i]==2;
            r[2*S+i]=g.board.terr[i]==1;
            r[3*S+i]=g.board.terr[i]==2;
            r[4*S+i]=(g.player()==1);
        }
        return out;
    }
};


// =========================
// AlphaZero-style MCTS
// =========================
struct MCTS {
    struct Node {
        int parent = -1;
        int parent_move = -1;

        float prior = 0.0f;
        float value_sum = 0.0f;
        int visits = 0;

        bool expanded = false;
        bool terminal = false;
        int winner = 0; // 0 none/tie, 1/2

        vector<int> moves;
        vector<int> child;
    };

    float cpuct = 1.5f;
    float dir_alpha = 0.3f;
    float dir_eps = 0.25f;
    int n_sims = 200;

    mt19937_64 rng;
    vector<Node> nodes;

    explicit MCTS(float cpuct_=1.5f, int n_sims_=200, float dir_alpha_=0.3f, float dir_eps_=0.25f, uint64_t seed=12345)
        : cpuct(cpuct_), dir_alpha(dir_alpha_), dir_eps(dir_eps_), n_sims(n_sims_), rng(seed) {}

    void reset_tree() { nodes.clear(); }

    using EvalFn = std::function<std::pair<py::array, float>(py::array)>;

    int new_node(int parent=-1, int parent_move=-1, float prior=0.0f) {
        nodes.push_back(Node{});
        int id = (int)nodes.size()-1;
        nodes[id].parent = parent;
        nodes[id].parent_move = parent_move;
        nodes[id].prior = prior;
        return id;
    }

    int select_child_ucb(int nid) {
        Node &n = nodes[nid];
        float best = -1e30f;
        int best_i = -1;
        float sqrt_vis = std::sqrt((float)std::max(1, n.visits));

        for (int i=0;i<(int)n.child.size();i++){
            int cid = n.child[i];
            Node &c = nodes[cid];

            float q = (c.visits==0) ? 0.0f : (c.value_sum / (float)c.visits);
            float u = cpuct * c.prior * (sqrt_vis / (1.0f + (float)c.visits));
            float score = q + u;

            if (score > best) { best = score; best_i = i; }
        }
        return best_i;
    }

    void expand_node(int nid, const vector<int> &legal_moves,
                     const vector<float> &priors, bool add_dirichlet) {
        Node &n = nodes[nid];
        n.expanded = true;

        n.moves.clear();
        n.child.clear();
        n.moves.reserve(legal_moves.size());
        n.child.reserve(legal_moves.size());

        vector<float> p;
        p.reserve(legal_moves.size());

        float sum = 0.0f;
        for (int mv : legal_moves) {
            float pr = (mv >= 0 && mv < (int)priors.size()) ? priors[mv] : 0.0f;
            pr = std::max(0.0f, pr);
            p.push_back(pr);
            sum += pr;
        }

        if (sum <= 1e-12f) {
            float uni = 1.0f / std::max(1, (int)legal_moves.size());
            for (auto &x : p) x = uni;
        } else {
            for (auto &x : p) x /= sum;
        }

        if (add_dirichlet && !legal_moves.empty()) {
            std::gamma_distribution<float> gamma(dir_alpha, 1.0f);
            vector<float> noise(legal_moves.size());
            float ns = 0.0f;
            for (size_t i=0;i<noise.size();i++){ noise[i]=gamma(rng); ns += noise[i]; }
            if (ns > 1e-12f) for (auto &x: noise) x /= ns;

            for (size_t i=0;i<p.size();i++){
                p[i] = (1.0f - dir_eps)*p[i] + dir_eps*noise[i];
            }
        }

        for (size_t i=0;i<legal_moves.size();i++){
            int mv = legal_moves[i];
            int cid = new_node(nid, mv, p[i]);
            n.moves.push_back(mv);
            n.child.push_back(cid);
        }
    }

    void backup(int nid, float value_from_current_player) {
        float v = value_from_current_player;
        int cur = nid;
        while (cur != -1) {
            Node &n = nodes[cur];
            n.visits += 1;
            n.value_sum += v;
            v = -v;
            cur = n.parent;
        }
    }

    pair<vector<float>, int> run(PyGame &game, const EvalFn &eval_fn) {
        reset_tree();
        nodes.reserve(n_sims * 2 + 5);
        int root = new_node(-1, -1, 1.0f);

        // Expand root
        {
            if (game.is_over()) {
                nodes[root].terminal = true;
                nodes[root].winner = game.winner();
            } else {
                py::gil_scoped_acquire gil;
                py::array st = game.state_tensor();
                auto [py_priors, v] = eval_fn(st);

                auto buf = py_priors.request();
                float *ptr = (float*)buf.ptr;
                vector<float> priors(game.n2(), 0.0f);
                for (int i = 0; i < game.n2(); i++) priors[i] = ptr[i];

                auto legal = game.legal_moves_all();
                expand_node(root, legal, priors, true);
                nodes[root].visits = 1;
                nodes[root].value_sum = v;
            }
        }

        for (int sim = 0; sim < n_sims; sim++) {
            int nid = root;
            int applied = 0;

            while (nodes[nid].expanded && !nodes[nid].terminal) {
                int ci = select_child_ucb(nid);
                if (ci < 0) break;

                int child = nodes[nid].child[ci];
                int mv = nodes[child].parent_move;

                int r = game.apply(mv);
                if (r != 0) {
                    nodes[child].terminal = true;
                    break;
                }

                nid = child;
                applied++;
            }

            float leaf_value = 0.0f;

            if (game.is_over()) {
                int w = game.winner();
                int stm = game.current_player();

                if (w == 0) leaf_value = 0.0f;
                else if (w == stm) leaf_value = -1.0f;
                else leaf_value = 1.0f;

                nodes[nid].terminal = true;
                nodes[nid].winner = w;
            } else {
                py::gil_scoped_acquire gil;
                py::array st = game.state_tensor();
                auto [py_priors, v] = eval_fn(st);

                auto buf = py_priors.request();
                float *ptr = (float*)buf.ptr;

                vector<float> priors(game.n2(), 0.0f);
                for (int i = 0; i < game.n2(); i++) priors[i] = ptr[i];

                auto legal = game.legal_moves_all();
                expand_node(nid, legal, priors, false);

                leaf_value = v;
            }

            backup(nid, leaf_value);

            for (int i = 0; i < applied; i++) game.undo();
        }

        vector<float> policy(game.n2(), 0.0f);
        int best_move = -1;
        int best_visits = -1;

        Node &r = nodes[root];
        for (size_t i = 0; i < r.child.size(); i++) {
            int cid = r.child[i];
            int mv = r.moves[i];
            int v = nodes[cid].visits;
            policy[mv] = (float)v;
            if (v > best_visits) {
                best_visits = v;
                best_move = mv;
            }
        }

        float sum = 0.0f;
        for (float x : policy) sum += x;
        if (sum > 0) for (float &x : policy) x /= sum;

        return {policy, best_move};
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
        .def("apply", &PyGame::apply)
        .def("undo", &PyGame::undo)
        .def("legal_moves_top20", &PyGame::legal_moves_top20)
        .def("legal_moves_all", &PyGame::legal_moves_all)
        .def("hash", &PyGame::hash)
        .def("state_tensor", &PyGame::state_tensor);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<float,int,float,float,uint64_t>(),
             py::arg("cpuct")=1.5f,
             py::arg("n_sims")=200,
             py::arg("dir_alpha")=0.3f,
             py::arg("dir_eps")=0.25f,
             py::arg("seed")=12345)
        .def("run", &MCTS::run,
             py::arg("game"),
             py::arg("eval_fn"));
}